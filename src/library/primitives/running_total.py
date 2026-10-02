import pandas as pd

def running_total(df, running_total_cols, group_by_cols=None):
    """
    Calculates a cumulative sum (running total) for specified numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    running_total_cols : list of str
        A list of numeric column names for which to calculate the running total.
    group_by_cols : list of str, optional
        A list of column names to group by. If provided, the running total
        resets for each group. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with additional columns for the running totals.
        New column names will be in the format '{original_col}_running_total'.
    """
    df_copy = df.copy()

    for col in running_total_cols:
        new_col_name = f"{col}_running_total"
        
        if group_by_cols:
            # Performs the cumulative sum within each defined group
            df_copy[new_col_name] = df_copy.groupby(group_by_cols)[col].cumsum()
        else:
            # Performs a global cumulative sum across the entire column
            df_copy[new_col_name] = df_copy[col].cumsum()

    return df_copy