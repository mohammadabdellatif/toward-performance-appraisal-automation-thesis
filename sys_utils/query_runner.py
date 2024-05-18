import pandas as pd
from sqlalchemy import text

from ds_extractor.commons import logger
from sys_utils.systemlogger import debug_df


def run_query(conn, query: tuple, index_col=["id"]):
    logger.info(f" query {query[0]}")
    df = pd.read_sql_query(text(query[1]), conn, index_col=index_col)
    logger.info(df.columns)
    debug_df(df)
    return df
