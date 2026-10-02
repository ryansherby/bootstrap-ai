def union(df, *other_dfs, align_by_name=True, output_order=None):
    """
    Stacks multiple data streams (pandas DataFrames) vertically.

    Parameters
    ----------
    df : pd.DataFrame
        The first DataFrame to be included in the union.
    *other_dfs : tuple of pd.DataFrame
        Additional DataFrames to stack.
    align_by_name : bool, optional
        If True, columns are aligned by their names. Missing columns will be
        filled with NaN. If False, columns are aligned by their position.
        In this mode, all DataFrames must have the same number of columns,
        and subsequent DataFrames' columns will be renamed to match the
        first DataFrame's column names. Defaults to True.
    output_order : list of int, optional
        A list of integers specifying the desired order of input streams
        in the output. The index corresponds to the position in the input
        (0 for `df`, 1 for `other_dfs[0]`, etc.). For example, `[1, 0]`
        would place the first DataFrame from `other_dfs` before `df`.
        If None, the original input order is maintained. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing all input DataFrames stacked vertically.

    Raises
    ------
    ValueError
        If `align_by_name` is False and DataFrames have different numbers
        of columns.
        If `output_order` contains invalid indices or does not account for
        all input DataFrames.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'C': [5, 6], 'A': [7, 8]})
    >>> df3 = pd.DataFrame({'B': [9, 10], 'D': [11, 12]})

    >>> # Align by name (default behavior)
    >>> result_by_name = union(df1, df2, df3)
    >>> print(result_by_name)
       A    B    C     D
    0  1.0  3.0  NaN   NaN
    1  2.0  4.0  NaN   NaN
    2  7.0  NaN  5.0   NaN
    3  8.0  NaN  6.0   NaN
    4  NaN  9.0  NaN  11.0
    5  NaN 10.0  NaN  12.0

    >>> # Align by position (requires same number of columns)
    >>> df_pos1 = pd.DataFrame({'Col1': [1, 2], 'Col2': [3, 4]})
    >>> df_pos2 = pd.DataFrame({'X': [5, 6], 'Y': [7, 8]})
    >>> result_by_position = union(df_pos1, df_pos2, align_by_name=False)
    >>> print(result_by_position)
       Col1  Col2
    0     1     3
    1     2     4
    2     5     7
    3     6     8

    >>> # Custom output order
    >>> result_ordered = union(df1, df2, output_order=[1, 0])
    >>> print(result_ordered)
       C    A    B
    0  5.0  7.0  NaN
    1  6.0  8.0  NaN
    2  NaN  1.0  3.0
    3  NaN  2.0  4.0
    """
    all_dfs = [df.copy()] + [d.copy() for d in other_dfs]

    if output_order is not None:
        if not all(0 <= i < len(all_dfs) for i in output_order):
            raise ValueError("Invalid index in output_order. Indices must be within the range of input DataFrames.")
        if len(output_order) != len(all_dfs):
            raise ValueError("output_order must contain an index for each input DataFrame.")
        reordered_dfs = [all_dfs[i] for i in output_order]
        all_dfs = reordered_dfs

    if not align_by_name:
        first_df_cols = all_dfs[0].columns.tolist()
        for i, current_df in enumerate(all_dfs):
            if len(current_df.columns) != len(first_df_cols):
                raise ValueError(
                    f"When align_by_name is False, all DataFrames must have the same number of columns. "
                    f"DataFrame at index {i} has {len(current_df.columns)} columns, "
                    f"while the first DataFrame has {len(first_df_cols)} columns."
                )
            if i > 0: # Rename columns of subsequent DataFrames
                current_df.columns = first_df_cols

    result_df = pd.concat(all_dfs, ignore_index=True, sort=False)

    return result_df