import pandas as pd
from pandas import DataFrame, Series
from sqlalchemy import Connection

import preprocessing.issues
from ds_extractor.commons import DatasetExtractorBase, Queries
from ds_extractor.wf_columns import statuses_as_cols
from sys_utils.query_runner import run_query
from sys_utils.systemlogger import SystemLogger

logger = SystemLogger.logger("issues_extractor")


class IssueHistoryReplay:

    def __init__(self,
                 issue_id,
                 issue: Series,
                 issue_history: DataFrame,
                 snapshot_cols: list[str],
                 record_preprocess: object = None,
                 separate_snapshots: bool = True):
        super().__init__()
        self.separate_snapshots = separate_snapshots
        self.issue_history = issue_history
        self.issue = issue
        self.issue_id = issue_id
        self.issues_snapshot = DataFrame(columns=snapshot_cols)
        self.last_status = None
        self.last_status_time = None
        self.turn = 1

        def noup(copy: Series):
            return copy

        self.record_preprocess = noup if record_preprocess is None else record_preprocess

    def replay(self):
        copy = self.__new_row_copy()

        self.last_status = "open"
        self.last_status_time = self.issue['issue_created']
        copy['started'] = self.issue['issue_created']
        copy['issue_assignee'] = None
        for (hidx, hist) in self.issue_history.iterrows():
            copy = self.__process_history_record(copy, hist)

        self.__close_last_record(copy)
        return self.issues_snapshot

    def __close_last_record(self, copy):
        copy['issueid'] = self.issue_id
        copy['ended'] = self.__last_record_end_time()
        if self.separate_snapshots:
            copy['turn'] = self.turn
        self.__append_row_to_ds(copy)

    def __last_record_end_time(self):
        return self.last_status_time \
            if (pd.isna(self.issue['issue_resolution_date']) or
                self.issue['issue_resolution_date'] < self.last_status_time) \
            else self.issue['issue_resolution_date']

    def __process_history_record(self, copy, hist):
        field = hist['field']
        value = hist['value']
        hist_date = hist['created']
        if field == 'assignee':
            if self.__is_assignee_changed(copy, value):
                copy = self.__calculate_wf_and_close_snapshot(copy, hist, hist_date)
            copy['issue_assignee'] = value
        if self.__is_wf_status_change(field, value):
            self.__update_snapshot_wf_status(copy, hist)
            self.last_status = value
            self.last_status_time = hist_date
        return copy

    def __calculate_wf_and_close_snapshot(self, copy, hist, hist_date):
        # an important note, when the assignee is being switched, the current status is counted twice to close the count
        # for previous assignee and count over with the current assignee, so if the issue was in the in progress then
        # switch from user a to b, we need to count a step for user a and a new step for user b as both worked on the
        # same step which should be counted
        self.__update_snapshot_wf_status(copy, hist)
        self.__close_prev_assignee_record(copy, hist)
        if self.separate_snapshots:
            copy = self.__new_row_copy()
            copy['started'] = hist_date
        self.last_status_time = hist_date
        return copy

    def __is_assignee_changed(self, copy, value):
        return pd.notna(copy['issue_assignee']) and copy['issue_assignee'] != value

    def __new_row_copy(self):
        copy = self.issue.copy()
        for c in self.issues_snapshot.columns:
            if "wfe_" in c:
                copy[c] = 0
            if "wf_" in c:
                copy[c] = None
        return copy

    def __update_snapshot_wf_status(self, copy, hist):
        status_time = (hist['created'] - self.last_status_time).total_seconds()
        wf_status = 'wf_' + self.last_status
        wfe_status = 'wfe_' + self.last_status
        if copy[wf_status] is None:
            copy[wf_status] = 0
        copy[wf_status] = copy[wf_status] + status_time
        copy[wfe_status] = copy[wfe_status] + 1

    def __is_wf_status_change(self, field, value):
        return field == 'status' and value != self.last_status

    def __close_prev_assignee_record(self, copy, hist):
        copy['issueid'] = self.issue_id
        copy['ended'] = hist['created']
        if self.separate_snapshots:
            copy['turn'] = self.turn
            self.turn += 1
            self.__append_row_to_ds(copy)

    def __append_row_to_ds(self, copy):
        copy = self.record_preprocess(copy)
        single_row = DataFrame(columns=self.issues_snapshot.columns, data=[copy])
        self.issues_snapshot = pd.concat([self.issues_snapshot, single_row], ignore_index=True)


class HistoryReplayIssuesDatasetExtractor(DatasetExtractorBase):
    def __init__(self,
                 db_url: str,
                 target_dir: str = ".",
                 issues_filename: str = "issues_snapshots.csv",
                 separate_snapshots: bool = True,
                 issues_preprocess: object = None) -> None:
        super().__init__(db_url, self.__extract_issues_snapshots, target_dir)
        self.issues_filename = issues_filename
        self.separate_snapshots = separate_snapshots

        def noup(ds):
            return ds

        self.issues_preprocess = noup if issues_preprocess is None else issues_preprocess

    def __extract_issues_snapshots(self, conn: Connection):
        issues_df = run_query(conn, Queries.issues(), index_col='id')
        issues_hist_df = run_query(conn, Queries.issues_change_history(), index_col='id')
        comments_summary_df = run_query(conn, Queries.comments_summary())

        cols = self.__build_dataset_columns(conn, issues_df)
        issues_snapshot = self.__build_snapshots_dataset(cols, issues_df, issues_hist_df, comments_summary_df)
        issues_snapshot = self.issues_preprocess(issues_snapshot)
        return [(issues_snapshot, self.issues_filename),
                (issues_hist_df, "issues_change_history.csv")]

    def __build_snapshots_dataset(self, cols: list,
                                  issues_df: DataFrame,
                                  issues_hist_df: DataFrame,
                                  comments_summary_df: DataFrame):
        issues_snapshot = DataFrame(columns=cols)
        total = len(issues_df)
        count = 1

        def update_comments_count(copy: pd.Series):
            snapshot_comments = comments_summary_df[
                (comments_summary_df['issueid'] == copy['issueid']) &
                comments_summary_df['created'].between(copy['started'], copy['ended'], inclusive='left')]
            copy['issue_comments_count'] = len(snapshot_comments)
            return copy

        for (idx, row) in issues_df.iterrows():
            issue_snapshot = self.__build_issue_snapshots(cols, idx, issues_hist_df, row, update_comments_count)
            issues_snapshot = pd.concat([issues_snapshot, issue_snapshot], ignore_index=True)
            logger.progress(f'processed %i of %i issues' % (count, total))
            count += 1
        logger.newline()
        logger.info("process completed")
        return issues_snapshot

    def __build_issue_snapshots(self, cols, idx, issues_hist_df, row, update_comments_count):
        issue_history = issues_hist_df[issues_hist_df["issueid"] == idx]
        issue_snapshot = IssueHistoryReplay(idx,
                                            row,
                                            issue_history,
                                            cols,
                                            record_preprocess=update_comments_count,
                                            separate_snapshots=self.separate_snapshots).replay()
        return issue_snapshot

    def __build_dataset_columns(self, conn, issues_df):
        wf_columns = self.__find_workflow_columns(conn)
        cols = ['issueid', 'started', 'ended']
        cols.extend(issues_df.columns.to_list())
        cols.extend(wf_columns)
        if self.separate_snapshots:
            cols.append('turn')
        return cols

    def __find_workflow_columns(self, conn):
        workflow_statuses = run_query(conn, Queries.possible_wf_statuses(), index_col='wf_status')
        wf_columns = statuses_as_cols(workflow_statuses)
        return wf_columns


class IssuesDatasetExtractor(HistoryReplayIssuesDatasetExtractor):

    def __init__(self, db_url: str,
                 target_dir: str = ".",
                 issues_filename: str = "issues.csv",
                 separate_snapshots: bool = False):
        super().__init__(db_url,
                         issues_filename=issues_filename,
                         separate_snapshots=separate_snapshots,
                         target_dir=target_dir,
                         issues_preprocess=self.__extra_preprocess)

    def __extra_preprocess(self, issue_df: DataFrame):
        issue_df = issue_df.rename(columns={'issueid': 'id'})

        if not self.separate_snapshots:
            issue_df.set_index('id', inplace=True)
        else:
            issue_df.rename_axis('idx', inplace=True)

        tpp = preprocessing.issues.TotalTimePreProcess()
        cc = preprocessing.issues.CommentsCountPreProcess()
        cntr = preprocessing.issues.ContributorsPreProcess()
        tps = preprocessing.issues.TotalProcessingStepsPreProcess()

        issue_df = tpp.pre_process(issue_df)
        issue_df = cc.pre_process(issue_df)
        issue_df = cntr.pre_process(issue_df)

        issue_df = tps.pre_process(issue_df)

        return issue_df


def extract_issues_in_study_scope(snapshots: DataFrame):
    types = ['Ticket', 'Deployment', 'HD Service']
    return snapshots[(snapshots['issue_proj'].str.match('\w{2}\d{2}\w{1,}'))
                     & (snapshots['issue_type'].isin(types))
                     & (snapshots['issue_created'] >= '2022-01-01')
                     & (snapshots['issue_created'] <= '2022-12-31')
                     & pd.notna(snapshots['issue_resolution_date'])]
