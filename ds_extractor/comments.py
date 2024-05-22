import math
import os.path
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from re import Pattern

import pandas as pd
from nltk import sent_tokenize
from pandas import DataFrame, Series

from ds_extractor.commons import DatasetExtractorBase, Queries
from sys_utils.query_runner import run_query
from sys_utils.systemlogger import SystemLogger

logger = SystemLogger.logger("comments")


def as_is(text): return text


split_rgx = re.compile(r"\s+")
word_regex = re.compile(r"(#[a-z_]+?#|[a-z0-9\-'\",\?!:\.#”“`;\(\)]+)[\.\?!,:]?")


class Replacement:
    def process(self, text: str):
        pass


class SentenceReplacement(Replacement):

    def __init__(self, nature: str, sentence_regex: str, val: str) -> None:
        self.val = val
        self.sentence_regex = re.compile(sentence_regex)
        self.nature = nature

    def process(self, text: str):
        if self.sentence_regex.fullmatch(text) is not None:
            return self.val
        return text


class GreedyReplacement:

    def __init__(self, replacement, replacements: list[Replacement]) -> None:
        self.replacements = replacements
        self.replacement = replacement

    def process(self, text: str):
        for r in self.replacements:
            if r.process(text) != text:
                return self.replacement
        return text


class RegexReplacement(Replacement):

    def __init__(self, nature, regex, replacement, sequential_join: bool = False):
        self.nature = nature
        self.regex = re.compile(regex)
        self.replacement = replacement

        self.sequential_join = as_is
        if sequential_join:
            self.sequential_join_regex = re.compile(f'(\s*%s\s*)+' % replacement)

            def join(text): return re.sub(self.sequential_join_regex, self.replacement, text).strip()

            self.sequential_join = join

    def process(self, text: str):
        processed = re.sub(self.regex, self.replacement, text).strip()
        return self.sequential_join(processed)


class TokenReplacement(Replacement):

    def __init__(self, val, tokens) -> None:
        self.val = val
        self.tokens: list[Pattern] = []
        t: str
        for t in tokens:
            if not re.compile(r'[\w-]{2,}').match(t):
                continue
            self.tokens.append(re.compile(r'\b' + t.strip() + r'\b'))

    def process(self, text: str):
        for t in self.tokens:
            text = re.sub(t, self.val, text)
        return text


def place_holder_token(token):
    return f'ph_{token}'


class CommentPreprocessor:

    def __init__(self, tokens_replacement: list[tuple] = []):
        self.p_regex = re.compile(r"(<p\s?[\w\W]*?>)(.+?)(</p>)")
        self.embedded_email_regex = re.compile(r'(.*to:.+subject:.+|.*from:.+subject:.+)')
        self.replacements = [
            RegexReplacement("Paragraph", r'</?p\s?[\w\W]*?>', ""),
            # RegexReplacement("whitespace", r'Â|â', ""),  # TODO review this NBSP character
            RegexReplacement("whitespace", r'\u00A0', " "),  # Replace non-breaking whitespace with a single space

            RegexReplacement("typed code block", r"{code(:\w+)?\}[\w\W\s\S]+\{code\}", place_holder_token('code'),
                             sequential_join=True),
            RegexReplacement("And", r'&amp;', 'and'),

            RegexReplacement("coloring", r"{color(:#\w+)?}", ""),

            RegexReplacement("user mentioned", r"\[\s*~[\w\.]+\s*\]", place_holder_token('user'), sequential_join=True),
            RegexReplacement("user mentioned", r"@([\w]+)(\s+[\w]+){,2}", place_holder_token('user'),
                             sequential_join=True),

            RegexReplacement("Break", r"<br\s?/>|</?p>|&lt;/?p&gt;", "\n"),
            RegexReplacement("JSON", r"{(\s*([\"']\w+[\"'])\s*:[\w\W\"']+,?)\s*}", place_holder_token('code'),
                             sequential_join=True),
            RegexReplacement("Escaped XML Tag", r"(&lt;?[\w:]+[\s\w\W]*?&gt;.+?)?(&lt;/[\w:]+&gt;)",
                             place_holder_token('code'), sequential_join=True),
            RegexReplacement("Escaped start XML Doc", r"&lt;\?xml[\s\w\W'\"]*?\?&gt;", place_holder_token('code'),
                             sequential_join=True),
            RegexReplacement("Escaped start XML Tag", r"&lt;?[\w:]+[\s\w\W'\"]*?&gt;", place_holder_token('code'),
                             sequential_join=True),
            RegexReplacement("Escaped end XML Tag", r"(&lt;/[\w:]+&gt;)", place_holder_token('code'),
                             sequential_join=True),
            RegexReplacement("Escaped self-closing XML Tag", r"(&lt;[\w:\s'\"]+/&gt;)", place_holder_token('code'),
                             sequential_join=True),
            RegexReplacement("HTML Tag", r"</?[\w]+[\s\w\W]*?>", ' '),

            RegexReplacement("Oracle Error", r"ora-\d+:\s+.+", place_holder_token('logs'),
                             sequential_join=True),
            GreedyReplacement(place_holder_token('logs'),
                              [RegexReplacement("Logs",
                                                r'\d{1,2}:\d{1,2}:\d{1,2}.+?\W(debug|warn|info|trace|error|severe)\W',
                                                place_holder_token('logs'), sequential_join=True),
                               RegexReplacement("Common exception log",
                                                r"\*+\s*exception\s+text\s*\*+.+", place_holder_token('dotnet_trace'),
                                                sequential_join=True),
                               RegexReplacement("stack trace",
                                                r"at\s+[\w\.$]+\s*(\([\w \d\.&,;:$\[\]`]+\)?|\(\s*\)?)",
                                                place_holder_token('stack_trace'),
                                                sequential_join=True),
                               RegexReplacement("caused by", r"caused by:\s+[\w\.$]+",
                                                place_holder_token('stack_trace'),
                                                sequential_join=True)]),
            RegexReplacement('tables', r'\|[\w\W]+\|', place_holder_token('info_table'), sequential_join=True),

            RegexReplacement("attachment", r"\[\^[\w\W]+?\]", place_holder_token('attachment'), sequential_join=True),

            RegexReplacement("Greetings", r"(dear|hi|hello)\s+@?[\w ]+\s*,", 'greetings '),
            RegexReplacement("greeting user", r"(dear|hi|hello)\s*ph_user\s*,?", 'greetings '),

            RegexReplacement("email", r"[\w._-]+[\w._-]@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+", place_holder_token('email')),
            # RegexReplacement("regards", r"[\w ]+?regards[,\sa-z-_]+", " regards"),

            RegexReplacement("Link", r"http[s]?://[\w\-\.\~\:/%]+(\?[\w%&=-]+)?", place_holder_token('link')),

            RegexReplacement("unclear bullet", r"(\s+\d{1,2}-)([a-z])", r"\1 \2"),

            RegexReplacement("IP address", r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", place_holder_token('ip_address')),
            # RegexReplacement("Digits", r"(\b)\d+(\.\d+)?(\b)", "\1ph_number\3"),

            RegexReplacement("special yes/no", r"\?y(es)?", "? yes"),
            RegexReplacement("special yes/no", r"\?n(o)?", "? no"),
            RegexReplacement("special yes/no", r"\?\s*y(es)?\n", "? yes. "),
            RegexReplacement("special yes/no", r"\?\s*n(o)?\n", "? no. "),
            RegexReplacement("special yes/no", r"\?\s*n(o)?$", "? no."),
            RegexReplacement("special yes/no", r"\?\s*y(es)?$", "? no."),

            # TODO have a look at this, this is a work around for known text structure
            # TODO adda sample for this
            RegexReplacement("admin comment replace", r"(\w)\s*(\d{1,2}-)\s*(did|will)", r"\1. \2 \3"),
            RegexReplacement("admin comment replace", r"([:?]\.)", r"\1 "),
            RegexReplacement("admin comment replace", r"\bh2\.?\b", r''),

            RegexReplacement("extra spaces", r"[\t\f\v ]{2,}", " "),
            RegexReplacement("extra spaces", r"[\n\r]{1,}", " ")
        ]

        self.post_tokenize_replacements = [
            RegexReplacement("stored procedure", r"begin [\w\W\s]+end;", place_holder_token('sql')),
            RegexReplacement("DDL",
                             r"(create|alter|drop)(or replace)?\s+(table|unique index|index|sequence|view|function|procedure|trigger|schema|session|package)[\s\w\W]+;?",
                             place_holder_token('sql')),
            RegexReplacement("create user",
                             r"create\s+user.+identified\s+by[\s\w\W]+;?",
                             place_holder_token('sql')),
            RegexReplacement("update statement",
                             # r"\s*update\s+\S+\s+set\s+(?:(?:\S+\s*=\s*(?:\S+|\S*\'.*?\'))\s*,\s*)*(?:\S+\s*=\s*(?:\S+|\S*\'.*?\'))\s+where\s+(?:(?:\S+\s*(?:=|>|<|<=|>=)\s*(?:\S+|\S*\'.*?\'))\s+(?:and|or)\s+)*(?:\S+\s*(?:=|>|<|<=|>=)\s*(?:\S+|\S*\'.*?\'));",
                             r'update(\s+.+\s+)(set)\s+[\w\W\s]+;?',
                             place_holder_token('sql')),
            RegexReplacement("insert statement",
                             # r"insert\s+into\s+\S+\s+\((?:\s*\S+\s*,?\s*)+\)\s+values\s+\((?:\s*(?:\S+|\'.*?\')\s*,?\s*)+\);",
                             r'insert\s+into\s+.+\svalues\s+.+;?',
                             place_holder_token('sql')),
            RegexReplacement("insert as select statement",
                             # r"insert\s+into\s+\S+\s+\((?:\s*\S+\s*,?\s*)+\)\s+values\s+\((?:\s*(?:\S+|\'.*?\')\s*,?\s*)+\);",
                             r'insert\s+into\s+.+(select\s+.+);?',
                             place_holder_token('sql')),
            RegexReplacement("delete statement",
                             r"delete\s+from\s+\w+(\.\w+)*(\s+where\s+.+)?;?",
                             place_holder_token('sql')),
            RegexReplacement("select statement",
                             # r"\s*(select)\s+.*?\s*(from)\s+.*?\s*(where\s+.*?\s*)?\s*(group\s+by\s+.*?\s*)?\s*(having\s+.*?\s*)?\s*(order\s+by\s+.*?\s*)?;?\s*",
                             r"(select)\s+(.*)?\s+(from)\s+(.*)?(\s+[\w\W\s]*?)?;?",
                             place_holder_token('sql')),
            RegexReplacement("commit statement", r"commit;", ''),
            RegexReplacement("setup scripts", r"(set|declare)[\s\w=\(\)]+;", place_holder_token('sql')),
            RegexReplacement("windows path", r'\b[a-z]?:[\\][^<>:\"/|?*]+\b', place_holder_token('path')),
            RegexReplacement("Posix path", r'\b/[\w/\-]{2,}\b', place_holder_token('path')),
            RegexReplacement("files", r"[\w\.-]+\.(sql|class|java|jsp|xml|asp|jar|war|zip|tar|rar|jks)",
                             place_holder_token('file'))
        ]
        for tp in tokens_replacement:
            self.post_tokenize_replacements.append(TokenReplacement(tp[0], tp[1]))

    def preprocess_comment(self, comment: str) -> list[str]:
        comment = comment.lower()

        paragraphs = self.p_regex.split(comment)
        pre_processed = []

        for paragraph in paragraphs:
            if re.search(self.embedded_email_regex, paragraph) is not None:
                pre_processed.append(place_holder_token('attached_email'))
                break
            self.__pre_process_paragraph(paragraph, pre_processed)
        return self.__tokenize(pre_processed)

    def __pre_process_paragraph(self, paragraph, pre_processed):
        content = paragraph.strip()
        for replacement in self.replacements:
            if content == '':
                return
            content = replacement.process(content)
        if len(pre_processed) == 0 or self.__is_same_as_prev_pre_processed(content, pre_processed):
            pre_processed.append(content)

    def __is_same_as_prev_pre_processed(self, content, pre_processed):
        return pre_processed[-1] != content

    def __tokenize(self, paragraphs: list[str]) -> list[str]:
        pre_processed_tokens = []
        for p in paragraphs:
            tokens = sent_tokenize(p)
            for replacement in self.post_tokenize_replacements:
                tokens = [self.__post_replacement(replacement, t) for t in tokens]
            pre_processed_tokens.extend(tokens)
        return pre_processed_tokens

    def __post_replacement(self, replacement: RegexReplacement, text: str):
        return replacement.process(text)
        # words = split_rgx.split(text)
        # for idx, b_word in enumerate(words):
        #     if not word_regex.fullmatch(b_word):
        #         words[idx] = place_holder_token('unknown')
        # return ' '.join(words)


class PreProcessResult:

    @staticmethod
    def success(utterances_df: DataFrame):
        return PreProcessResult(True, utterances_df, None)

    @staticmethod
    def failed(error: Exception):
        return PreProcessResult(False, None, error)

    def __init__(self, completed: bool, utterances_df: DataFrame, error: Exception):
        self.utterances_df = utterances_df
        self.completed = completed
        self.error = error


class CommentsDatasetExtractor(DatasetExtractorBase):

    def __init__(self, db_url: str, target_dir: str = ".", thread_count: int = 8):
        super().__init__(db_url, self.__extract_comments_dataset, target_dir)
        self.thread_count = thread_count

    def __extract_comments_dataset(self, conn):
        comments_df = run_query(conn, Queries.comments(), index_col='id')
        issue_description_df = run_query(conn, Queries.issues_description_as_comments(), index_col='id')
        comments_df = pd.concat([comments_df, issue_description_df])
        profiles_df = run_query(conn, Queries.users_profiles(), index_col='user_name')
        utterances_df = self.__preprocess_comments(comments_df, profiles_df)
        utterances_df.rename_axis('id', inplace=True)
        utterances_df.sort_values(['issueid', 'created'], inplace=True)
        self.__impute_comment_seq(utterances_df)
        logger.info("completed")
        return [(utterances_df, "utterances.csv")]

    def __impute_comment_seq(self, utterances_df):
        logger.info("impute comment sequence")
        utterances_df['comment_seq'] = 0
        c_seq = -1
        issue_id = None
        comment_id = None
        for idx, r in utterances_df.iterrows():
            if issue_id != r['issueid']:
                issue_id = r['issueid']
                c_seq = -1
            if comment_id != idx:
                comment_id = idx
                c_seq += 1
            utterances_df.loc[idx, 'comment_seq'] = c_seq

    def __preprocess_comments(self, comments_df: DataFrame, profiles_df: DataFrame):
        work_share = int(math.ceil(len(comments_df) / self.thread_count))
        logger.info("Number of worker threads %i will work on a share of %i items" % (self.thread_count, work_share))
        comment_preprocessor = self.__comments_preprocessor(profiles_df)
        status_listener = StatusPrinter(total_count=len(comments_df))

        results = self.__execute_tasks(comment_preprocessor,
                                       comments_df,
                                       status_listener,
                                       work_share)

        logger.info(f"results dataframes %i" % len(results))

        [print('failed with error' + str(r.error)) for r in results if not r.completed]
        return pd.concat([r.utterances_df for r in results if r.completed])

    def __execute_tasks(self, comment_preprocessor, comments_df, status_listener, work_share):
        results: [PreProcessResult] = []
        with ThreadPoolExecutor(max_workers=self.thread_count + 1) as executor:
            for task_id in range(0, self.thread_count):
                task = self.__new_task(comment_preprocessor,
                                       comments_df,
                                       status_listener.update,
                                       task_id,
                                       work_share)
                executor.submit(lambda: results.append(task.run()))
        return results

    def __comments_preprocessor(self, profiles_df) -> CommentPreprocessor:
        names: set[str] = self.__names_as_set(profiles_df)
        # the product names list should come from a file
        products = self.__products_names()
        return CommentPreprocessor(
            tokens_replacement=[(place_holder_token('user'), names), (place_holder_token('product'), products)])

    def __products_names(self):
        products: set[str] = set()
        self.__add_names_from_file('./temp_data/products.txt', products)
        return products

    def __new_task(self, comment_preprocessor, comments_df, status_listener, task_id, work_share):
        share_start = int(task_id * work_share)
        share_end = int((task_id + 1) * work_share)
        logger.info(f'share indexes %i-%i' % (share_start, share_end))
        the_comments = comments_df.iloc[share_start:share_end]
        return CommentsPreProcessTask(task_id, the_comments, comment_preprocessor, status_listener)

    def __names_as_set(self, profiles_df: DataFrame) -> set[str]:
        r: set[str] = set()
        for names in profiles_df.to_numpy():
            [r.add(str(name).strip().lower()) for name in names if self.__is_valid_name(name)]

        self.add_statis_names(r)

        return r

    def add_statis_names(self, names_list):
        # those should be read from a file
        self.__add_names_from_file("./temp_data/names.txt", names_list)

    def __add_names_from_file(self, file_path, names_list):
        static_names = Path(file_path)
        if os.path.exists(static_names):
            with open(static_names) as file:
                for line in file:
                    names_list.add(line.strip())

    def __is_valid_name(self, n: str):
        if self.__is_blank(n) \
                or self.__is_letters(n) \
                or self.__is_keyword(n):
            return False
        return True

    def __is_keyword(self, n):
        return n.strip() in ('helpdesk', 'support', 'it', 'test', 'web', 'channels', 'day', 'check')

    def __is_letters(self, n):
        return len(n) < 3

    def __is_blank(self, n):
        return n is None or n.strip() == ''


class StatusPrinter:

    def __init__(self, total_count: int) -> None:
        self.total_count = total_count
        self.total_processed = 0

    def update(self, status: bool, extra: object):
        if status:
            self.total_processed = self.total_processed + extra
            sys.stdout.write(f"\rprocessed %i comment out of %i" % (self.total_processed, self.total_count))
            sys.stdout.flush()
        else:
            logger.error("One of the threads failed", extra)


class CommentsPreProcessTask:

    def __init__(self,
                 thread_id: int,
                 df: DataFrame,
                 comment_preprocessor: CommentPreprocessor,
                 status_listener: object) -> None:
        super().__init__()
        self.comment_preprocessor = comment_preprocessor
        self.status_listener = status_listener
        self.df = df
        self.thread_id = thread_id

    def run(self) -> PreProcessResult:
        logger.info(f"thread %i is started to process %i records" % (self.thread_id, len(self.df)))
        utr_cols = [f for f in self.df.columns.array]
        utr_cols.insert(-1, 'utr_seq')
        utterances_df = DataFrame(columns=utr_cols)
        try:
            row: Series
            count = 0
            for (i, row) in self.df.iterrows():
                utterances_df = self.__preprocess_comment_row(row, utr_cols, utterances_df)
                count += 1
                self.status_listener(True, 1)
        except Exception as e:
            self.status_listener(False, e)
            return PreProcessResult.failed(error=e)
        else:
            return PreProcessResult.success(utterances_df=utterances_df)

    def __preprocess_comment_row(self, row, utr_cols, utterances_df):
        comment = row['actionbody']
        utterances = self.comment_preprocessor.preprocess_comment(comment)
        utr_seq = 0
        for utterance in utterances:
            if utterance is None or utterance.strip() == '':
                continue
            utterances_df = self.__append_utterance_to_df(row, utr_cols, utr_seq, utterance, utterances_df)
            utr_seq += 1
        return utterances_df

    def __append_utterance_to_df(self, row, utr_cols, utr_seq, utterance, utterances_df):
        new_row: Series = row.copy()
        new_row['utr_seq'] = utr_seq
        new_row['actionbody'] = utterance
        return pd.concat([utterances_df, DataFrame(columns=utr_cols, data=[new_row])])

# if __name__ == '__main__':
#     static_names = set()
#     CommentsDatasetExtractor.add_statis_names(static_names)
#     cp = CommentPreprocessor(tokens_replacement=[('ph_user', static_names)])
#     c = cp.preprocess_comment(
#         "<p>Dear <span>[~jo.01]</span> </p><p>Kindly be informed that your ticket has been assigned.</p><p>Regards,</p><p>HelpDesk Team.</p>")
#     print(c)
