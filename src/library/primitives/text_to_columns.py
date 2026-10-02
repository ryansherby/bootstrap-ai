def text_to_columns(df, column_to_split, delimiters, split_type, num_columns=None):
    """
    Splits a single string field into multiple columns or rows based on specified delimiters.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column_to_split : str
        The name of the column to split.
    delimiters : str
        Characters to split on (e.g., ",;\n"). Multiple characters will be treated
        as individual delimiters, and the column will be split by any of them.
    split_type : str
        Type of split: 'to_columns' to create new columns, or 'to_rows' to expand
        into multiple rows.
    num_columns : int, optional
        Required if `split_type` is 'to_columns'. Specifies the maximum number
        of new columns to create. If the split results in fewer parts, the
        remaining columns will be filled with NaN. If more parts, only the
        first `num_columns` will be kept.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified column split.

    Raises
    ------
    ValueError
        If `split_type` is invalid or `num_columns` is not provided for 'to_columns'.

    Examples
    --------
    >>> data = {'id': [1, 2], 'text_data': ['apple,banana,cherry', 'grape;orange']}
    >>> df = pd.DataFrame(data)
    >>> # Split 'text_data' into new columns
    >>> df_cols = text_to_columns(df, 'text_data', ',', 'to_columns', num_columns=3)
    >>> print(df_cols)
       id text_data_split_0 text_data_split_1 text_data_split_2
    0   1             apple            banana            cherry
    1   2       grape;orange               NaN               NaN
    >>> # Split 'text_data' into new rows
    >>> df_rows = text_to_columns(df, 'text_data', ',', 'to_rows')
    >>> print(df_rows)
       id    text_data
    0   1        apple
    0   1       banana
    0   1       cherry
    1   2  grape;orange
    >>> # Split 'text_data' with multiple delimiters
    >>> data_multi = {'id': [1, 2], 'text_data': ['apple,banana;cherry', 'grape\\norange']}
    >>> df_multi = pd.DataFrame(data_multi)
    >>> df_multi_split = text_to_columns(df_multi, 'text_data', ',;\\n', 'to_columns', num_columns=3)
    >>> print(df_multi_split)
       id text_data_split_0 text_data_split_1 text_data_split_2
    0   1             apple            banana            cherry
    1   2             grape            orange               NaN
    """
    modified_df = df.copy()

    if column_to_split not in modified_df.columns:
        raise ValueError(f"Column '{column_to_split}' not found in DataFrame.")

    if not pd.api.types.is_string_dtype(modified_df[column_to_split]):
        raise TypeError(f"Column '{column_to_split}' must be of string type.")

    if not delimiters:
        raise ValueError("Delimiters string cannot be empty.")

    canonical_delimiter = delimiters[0]
    temp_column = modified_df[column_to_split].astype(str)

    for delim in delimiters:
        if delim != canonical_delimiter:
            temp_column = temp_column.str.replace(delim, canonical_delimiter)

    if split_type == 'to_columns':
        if num_columns is None:
            raise ValueError("`num_columns` must be provided when `split_type` is 'to_columns'.")
        if not isinstance(num_columns, int) or num_columns <= 0:
            raise ValueError("`num_columns` must be a positive integer.")

        split_data = temp_column.str.split(canonical_delimiter, expand=True)

        if split_data.shape[1] < num_columns:
            for i in range(split_data.shape[1], num_columns):
                split_data[i] = np.nan
        
        split_data = split_data.iloc[:, :num_columns]

        new_column_names = [f"{column_to_split}_split_{i}" for i in range(num_columns)]
        split_data.columns = new_column_names

        modified_df = modified_df.drop(columns=[column_to_split])
        modified_df = pd.concat([modified_df, split_data], axis=1)

    elif split_type == 'to_rows':
        modified_df[column_to_split] = temp_column.str.split(canonical_delimiter)
        modified_df = modified_df.explode(column_to_split)
        modified_df = modified_df.reset_index(drop=True)

    else:
        raise ValueError(f"Invalid `split_type`: '{split_type}'. Must be 'to_columns' or 'to_rows'.")

    return modified_df