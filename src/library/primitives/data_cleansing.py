def data_cleansing(
    df,
    columns_to_clean,
    replace_nulls_with_blanks=True,
    replace_nulls_with_zeros=True,
    remove_leading_trailing_whitespace=True,
    remove_tabs=True,
    remove_line_breaks=True,
    remove_numbers=False,
    remove_punctuation=False
):
    """
    Performs bulk cleanup of nulls, whitespace, and unwanted characters in specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    columns_to_clean : list of str
        A list of column names to apply cleansing operations to.
    replace_nulls_with_blanks : bool, optional
        If True, replaces nulls in string columns with empty strings. Defaults to True.
    replace_nulls_with_zeros : bool, optional
        If True, replaces nulls in numeric columns with 0. Defaults to True.
    remove_leading_trailing_whitespace : bool, optional
        If True, removes leading and trailing whitespace from string columns. Defaults to True.
    remove_tabs : bool, optional
        If True, removes tab characters from string columns. Defaults to True.
    remove_line_breaks : bool, optional
        If True, removes line break characters (\n, \r) from string columns. Defaults to True.
    remove_numbers : bool, optional
        If True, removes all numeric digits from string columns. Defaults to False.
    remove_punctuation : bool, optional
        If True, removes all punctuation characters from string columns. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified cleansing operations applied.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {
    ...     'col_str': ['  hello world  ', '  test\tdata\n', None, '123abc!'],
    ...     'col_num': [1, 2, None, 4],
    ...     'col_other': ['keep', 'this', 'unchanged', 'data']
    ... }
    >>> df = pd.DataFrame(data)
    >>> cleaned_df = data_cleansing(
    ...     df,
    ...     columns_to_clean=['col_str', 'col_num'],
    ...     replace_nulls_with_blanks=True,
    ...     replace_nulls_with_zeros=True,
    ...     remove_leading_trailing_whitespace=True,
    ...     remove_tabs=True,
    ...     remove_line_breaks=True,
    ...     remove_numbers=True,
    ...     remove_punctuation=True
    ... )
    >>> print(cleaned_df['col_str'].tolist())
    ['hello world', 'testdata', '', 'abc']
    >>> print(cleaned_df['col_num'].tolist())
    [1.0, 2.0, 0.0, 4.0]
    """
    cleaned_df = df.copy()

    for col in columns_to_clean:
        if col not in cleaned_df.columns:
            continue

        # Handle nulls
        if cleaned_df[col].dtype == 'object': # String column
            if replace_nulls_with_blanks:
                cleaned_df[col] = cleaned_df[col].fillna('')
        elif pd.api.types.is_numeric_dtype(cleaned_df[col]): # Numeric column
            if replace_nulls_with_zeros:
                cleaned_df[col] = cleaned_df[col].fillna(0)

        # Apply string cleaning operations if column is of object type
        if cleaned_df[col].dtype == 'object':
            # Ensure column is string type for string operations
            cleaned_df[col] = cleaned_df[col].astype(str)

            if remove_leading_trailing_whitespace:
                cleaned_df[col] = cleaned_df[col].str.strip()
            if remove_tabs:
                cleaned_df[col] = cleaned_df[col].str.replace('\t', '')
            if remove_line_breaks:
                cleaned_df[col] = cleaned_df[col].str.replace('\n', '').str.replace('\r', '')
            if remove_numbers:
                cleaned_df[col] = cleaned_df[col].str.replace(r'\d+', '', regex=True)
            if remove_punctuation:
                # This regex removes any character that is not a word character (alphanumeric + underscore) or whitespace.
                # This effectively removes most common punctuation while preserving internal spaces.
                cleaned_df[col] = cleaned_df[col].str.replace(r'[^\w\s]', '', regex=True)
                # After removing punctuation, normalize internal spaces and re-strip in case punctuation was at ends.
                cleaned_df[col] = cleaned_df[col].str.strip().str.replace(r'\s+', ' ', regex=True)

    return cleaned_df