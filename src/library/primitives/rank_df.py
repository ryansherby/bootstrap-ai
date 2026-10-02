def rank_df(df, rank_field, rank_order='ascending', group_by=None, rank_method='competition', output_field_name='rank'):
    """
    Assigns a ranking value (1, 2, 3...) to records based on a specific numeric or alphanumeric field.
    It is often used to find "Top N" items or to order records within specific groups.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    rank_field : str
        The name of the column to be ranked.
    rank_order : str, optional
        The order of ranking. 'ascending' (1 = smallest value) or 'descending' (1 = largest value).
        Defaults to 'ascending'.
    group_by : list of str, optional
        A list of column names to group by. If provided, ranking will restart within each group.
        Defaults to None.
    rank_method : str, optional
        The method used to handle ties:
        - 'competition' (1224): Items that tie receive the same rank, and a gap is left in the sequence.
        - 'dense' (1223): Items that tie receive the same rank, but the next rank follows immediately without a gap.
        - 'ordinal' (1234): Every item receives a unique rank based on its position, even if values are identical.
        - 'modified_competition' (1334): Similar to Competition, but the tied items receive the highest possible rank in that group.
        Defaults to 'competition'.
    output_field_name : str, optional
        The name for the new rank column. Defaults to 'rank'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the added rank column.

    Examples
    --------
    >>> data = {'category': ['A', 'A', 'B', 'B', 'A', 'B'],
    ...         'value': [10, 20, 15, 15, 10, 30]}
    >>> df = pd.DataFrame(data)
    >>> ranked_df = rank_df(df, rank_field='value', rank_order='ascending', group_by=['category'], rank_method='dense', output_field_name='category_rank')
    >>> print(ranked_df)
      category  value  category_rank
    0        A     10            1.0
    1        A     20            2.0
    2        B     15            1.0
    3        B     15            1.0
    4        A     10            1.0
    5        B     30            2.0
    """
    df_copy = df.copy()

    ascending_bool = True if rank_order == 'ascending' else False

    pandas_rank_method_map = {
        'competition': 'min',
        'dense': 'dense',
        'ordinal': 'first',
        'modified_competition': 'max'
    }
    method = pandas_rank_method_map.get(rank_method, 'min') # Default to 'min' for safety

    if group_by:
        df_copy[output_field_name] = df_copy.groupby(group_by)[rank_field].rank(
            method=method, ascending=ascending_bool
        )
    else:
        df_copy[output_field_name] = df_copy[rank_field].rank(
            method=method, ascending=ascending_bool
        )

    return df_copy