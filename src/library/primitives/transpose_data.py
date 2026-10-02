def transpose_data(df, key_fields, data_fields):
    """
    Pivots "wide" data into "long" data by collapsing specified data columns into two new fields.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with "wide" data.
    key_fields : list of str
        A list of column names that should remain as identifier variables.
    data_fields : list of str
        A list of column names that should be unpivoted (collapsed) into new 'variable' and 'value' columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with data in "long" format.

    Examples
    --------
    >>> data = {'ID': [1, 2], 'Name': ['A', 'B'], 'Col1': [10, 20], 'Col2': [15, 25]}
    >>> df = pd.DataFrame(data)
    >>> df
       ID Name  Col1  Col2
    0   1    A    10    15
    1   2    B    20    25

    >>> key_fields = ['ID', 'Name']
    >>> data_fields = ['Col1', 'Col2']
    >>> long_df = transpose_data(df, key_fields, data_fields)
    >>> long_df
       ID Name variable  value
    0   1    A     Col1     10
    1   2    B     Col1     20
    2   1    A     Col2     15
    3   2    B     Col2     25
    """
    df_copy = df.copy()
    long_df = pd.melt(df_copy, id_vars=key_fields, value_vars=data_fields, var_name='variable', value_name='value')
    return long_df