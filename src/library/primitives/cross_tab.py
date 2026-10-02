def cross_tab(df, group_by_columns, column_headers_field, values_field, aggregation_method='sum'):
    """
    Pivots "Long" data into "Wide" data (Rows to Columns) based on specified fields and aggregation method.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame in long format.
    group_by_columns : list of str or str
        Column(s) to group data by, forming the new index of the pivoted table.
    column_headers_field : str
        The field whose unique values will become the new column headers.
    values_field : str
        The data field whose values will populate the new columns.
    aggregation_method : str, optional
        The method for aggregating values. Supported methods: 'sum', 'average', 'count', 'min', 'max', 'concatenate'.
        Defaults to 'sum'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame in wide format after pivoting.

    Examples
    --------
    >>> data = {
    ...     'Region': ['East', 'East', 'West', 'West', 'East'],
    ...     'Product': ['A', 'B', 'A', 'C', 'A'],
    ...     'Sales': [100, 150, 200, 50, 120],
    ...     'Notes': ['Good', 'OK', 'Great', 'Fine', 'Good']
    ... }
    >>> df = pd.DataFrame(data)
    >>> # Original DataFrame:
    >>> #   Region Product  Sales  Notes
    >>> # 0   East       A    100   Good
    >>> # 1   East       B    150     OK
    >>> # 2   West       A    200  Great
    >>> # 3   West       C     50   Fine
    >>> # 4   East       A    120   Good

    >>> # Example 1: Sum of Sales by Region, with Products as columns
    >>> pivoted_df_sum = cross_tab(df, group_by_columns='Region',
    ...                            column_headers_field='Product',
    ...                            values_field='Sales',
    ...                            aggregation_method='sum')
    >>> # Pivoted DataFrame (Sum of Sales):
    >>> # Product      A      B     C
    >>> # Region
    >>> # East     220.0  150.0   NaN
    >>> # West     200.0    NaN  50.0

    >>> # Example 2: Concatenate Notes by Region, with Products as columns
    >>> pivoted_df_concat = cross_tab(df, group_by_columns='Region',
    ...                                column_headers_field='Product',
    ...                                values_field='Notes',
    ...                                aggregation_method='concatenate')
    >>> # Pivoted DataFrame (Concatenated Notes):
    >>> # Product          A    B     C
    >>> # Region
    >>> # East     Good, Good   OK  None
    >>> # West          Great  None  Fine
    """
    df_copy = df.copy()

    agg_func_map = {
        'sum': np.sum,
        'average': np.mean,
        'count': 'count',
        'min': np.min,
        'max': np.max,
        'concatenate': lambda x: ', '.join(x.dropna().astype(str)) if not x.dropna().empty else None
    }

    if aggregation_method not in agg_func_map:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}. "
                         f"Supported methods are: {', '.join(agg_func_map.keys())}")

    agg_func = agg_func_map[aggregation_method]

    if isinstance(group_by_columns, str):
        group_by_columns = [group_by_columns]

    pivoted_df = pd.pivot_table(
        df_copy,
        index=group_by_columns,
        columns=column_headers_field,
        values=values_field,
        aggfunc=agg_func
    )

    if aggregation_method not in ['concatenate']:
        pivoted_df = pivoted_df.fillna(np.nan)

    return pivoted_df