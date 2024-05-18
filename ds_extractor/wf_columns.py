
def statuses_as_cols(wf_statuses_df):
    cols = []
    for (i, c) in wf_statuses_df.iterrows():
        cols.append("wf_" + i)
        cols.append("wfe_" + i)
    return cols
