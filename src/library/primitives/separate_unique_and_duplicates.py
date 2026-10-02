def separate_unique_and_duplicates(df, columns_to_evaluate):
    """
    Separates a DataFrame into two based on unique and duplicate records
    for a specified set of columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    columns_to_evaluate : list of str
        A list of column names to consider when identifying unique and duplicate records.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        A tuple containing two DataFrames:
        - unique_df (pd.DataFrame): Contains records that are unique based on the
          `columns_to_evaluate` (i.e., appear only once).
        - duplicates_df (pd.DataFrame): Contains all records that are part of a
          duplicate set based on the `columns_to_evaluate` (i.e., appear more than once).

    Examples
    --------
    >>> data = {'id': [1, 2, 1, 3, 2],
    ...         'value': ['A', 'B', 'A', 'C', 'B']}
    >>> df = pd.DataFrame(data)
    >>> unique_records, duplicate_records = separate_unique_and_duplicates(df, ['id', 'value'])
    >>> print("Unique Records:")
    >>> print(unique_records)
       id value
    3   3     C
    >>> print("Duplicate Records:")
    >>> print(duplicate_records)
       id value
    0   1     A
    1   2     B
    2   1     A
    4   2     B
    """
    df_copy = df.copy()

    # Identify all rows that are part of a duplicate set
    # keep=False marks all occurrences of a duplicate set as True
    is_duplicate = df_copy.duplicated(subset=columns_to_evaluate, keep=False)

    # Records that are truly unique (appear only once)
    unique_df = df_copy[~is_duplicate]

    # All records that are part of a duplicate set
    duplicates_df = df_copy[is_duplicate]

    return unique_df, duplicates_df