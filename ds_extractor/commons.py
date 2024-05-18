import os
from os.path import exists

from pandas import DataFrame
from sqlalchemy import create_engine, Connection

from sys_utils.systemlogger import SystemLogger

logger = SystemLogger.logger("commons")


class Queries:
    # TODO add index col name to the tuple
    @staticmethod
    def issues_assignment_history(): return ('issues_assignment_history', """
select ci.id
     , cg.issueid
     , ci.newvalue                     as assignee
     , cg.created                      as created
     , last_change_by_user.last_change as assignee_last_change
from changeitem ci
         join changegroup cg on ci.groupid = cg.id
         left join (select cg.issueid, cg.author, max(cg.created) as last_change
                    from changegroup cg
                    group by cg.issueid, cg.author, cg.issueid) as last_change_by_user
                   on cg.issueid = last_change_by_user.issueid and last_change_by_user.author = ci.newvalue
where ci.field = 'assignee'
  and ci.newvalue is not null
  and lower(ci.oldvalue) <> lower(ci.newvalue)
  and cg.id not in (select min(id) from changegroup group by issueid);
    """)

    @staticmethod
    def issues(): return ('list_issues', """
SELECT I.ID,
       I.ISSUENUM                                                  as issue_num,
       P.PKEY                                                      as issue_proj,
       I.REPORTER                                                  as issue_reporter,
       I.ASSIGNEE                                                  as issue_assignee,
       contr.issue_contr_count                                     as issue_contr_count,
       IT.PNAME                                                    AS ISSUE_TYPE,
       case when PR.PNAME is null then 'unknown' else pr.pname end AS ISSUE_PRIORITY,
       I.CREATED                                                   AS ISSUE_CREATED,
       I.RESOLUTIONDATE                                            AS ISSUE_RESOLUTION_DATE,
       r.pname                                                     AS ISSUE_RESOLUTION,
       regexp_replace(lower(stts.pname), '[( )(/)]+', '_', 'g')    as ISSUE_STATUS,
       cmt.cnt                                                     as issue_comments_count,
       last_change.last_change_date                                as last_change_date
FROM JIRAISSUE I
         left join (select count(*) as cnt, issueid
                    from jiraaction ac
                    where actiontype = 'comment'
                    group by issueid) cmt
                   on cmt.issueid = i.id
         left join (select ia.issueid, count(*) as issue_contr_count
                    from (select cg.issueid, lower(ci.newvalue)
                          from changeitem ci
                                   join changegroup cg on ci.groupid = cg.id
                          where ci.field = 'assignee'
                            and lower(ci.oldvalue) <> lower(ci.newvalue)
                            and ci.newvalue is not null
                          group by cg.issueid, lower(ci.newvalue)) as ia
                    group by ia.issueid) as contr on contr.issueid = i.id
         left join (select cg.issueid, max(cg.created) as last_change_date
                    from changegroup cg
                    group by cg.issueid) as last_change on last_change.issueid = I.id
         left join resolution r on r.id = i.resolution
         left join PRIORITY PR on pr.id = i.priority
         join PROJECT P on p.id = i.project
         join ISSUETYPE IT on it.id = i.issuetype
         join issuestatus stts on i.issuestatus = stts.id
""")

    @staticmethod
    def comments(): return ('list_issues_comments', """
select id,
       issueid,
       created,
       author,
       actionbody,
       case when rolelevel = 10301 then 1 else 0 end as is_private
from jiraaction
where actiontype = 'comment'
  and issueid in (select j.id from jiraissue j where j.created between '2021-12-31 23:59:59' and '2022-12-31 23:59:59'
   and j.RESOLUTIONDATE is not null)
order by issueid, created
    """)

    @staticmethod
    def issues_description_as_comments(): return ('list_issue_description', """
select max_id.i + row_number() over (order by id)                      id,
       id                                                           as issueid,
       j.created,
       j.reporter                                                   as author,
       (case when description is null then '' else description end) as actionbody,
       0                                                            as is_private
from jiraissue j,
     (select max(id) as i from jiraaction) max_id
where j.created between '2021-12-31 23:59:59' and '2022-12-31 23:59:59'
  and j.RESOLUTIONDATE is not null
    """)

    @staticmethod
    def users_profiles(): return ('list users profiles', """
    select lower_display_name,user_name, lower_first_name, lower_last_name
    from cwd_user
    """)

    @staticmethod
    def comments_summary(): return ('list_issues_comments_summary', """
    select id, issueid, created, author, actionnum
    from jiraaction
    where actiontype = 'comment'
    order by created desc
        """)

    @staticmethod
    def status_change(): return ('issue_status_change', """
    select ci.id,
       cg.issueid,
       cg.author,
       regexp_replace(lower(newstring), '[( )(/)]+', '_', 'g') as status,
       cg.created
from changeitem ci
         join changegroup cg on cg.id = ci.groupid
where field = 'status'
""")

    @staticmethod
    def possible_wf_statuses(): return ('possible_wf_statuses', """
select distinct regexp_replace(lower(newstring), '[( )(/)]+', '_', 'g') AS wf_status
from changeitem
where field = 'status'""")

    @staticmethod
    def issues_change_history(): return ('issues_change_history', """
select ci.id
     , cg.issueid
     , ci.field
     , case
           when ci.field = 'status'
               then regexp_replace(lower(ci.newstring), '[( )(/)]+', '_', 'g')
           else ci.newvalue
    end           as value
     , cg.created as created
     , cg.id      as change_group_id
from changeitem ci
         join changegroup cg on ci.groupid = cg.id
where ci.field in ('assignee', 'status')
order by cg.id
    """)


class DatasetExtractorBase:
    def __init__(self, db_url: str, processor: object, target_dir: str = ".") -> None:
        self.__db_url = db_url
        self.__target_dir = target_dir
        self.__processor = processor

    def process(self):
        alchemy_engine = create_engine(self.__db_url);
        conn: Connection
        with alchemy_engine.connect() as conn:
            datasets: list(tuple) = self.__processor(conn)
            for ds in [] if datasets is None else datasets:
                self.__write_df_as_csv(ds[0], ds[1])

    def __write_df_as_csv(self, df: DataFrame, file_name: str):
        if not exists(self.__target_dir):
            os.makedirs(self.__target_dir)
        df.to_csv(self.__target_dir + "/" + file_name)
