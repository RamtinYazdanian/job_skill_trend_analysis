import numpy as np
import pandas as pd


def get_period_of_time(df, start, end):
    if start is not None and end is not None:
        return df.loc[(df.Date >= start) & (df.Date < end)]
    elif start is None and end is not None:
        return df.loc[(df.Date < end)]
    elif start is not None and end is None:
        return df.loc[(df.Date >= start)]
    else:
        return df


def remove_list_nans(l):
    return [x for x in l if not pd.isna(x)]


def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res


def filter_out_by_common_index(df_ref, df_target, index_col='common_index'):
    return df_target.loc[df_target[index_col].apply(lambda x: x in df_ref[index_col].values)]