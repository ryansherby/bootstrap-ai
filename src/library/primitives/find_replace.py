def find_replace(df, column_to_search, find_value, search_criteria, replace_mode, replace_with_value=None, target_column_for_replace=None, lookup_df=None, lookup_key_column_lookup=None, columns_to_append=None):
    """
    Searches for a string in a specified column and either replaces it or appends data from a lookup DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column_to_search : str
        The name of the column in `df` to search for `find_value`.
    find_value : str
        The string value to search for.
    search_criteria : str
        The criteria for searching. Must be one of 'contains', 'starts_with', 'equals'.
    replace_mode : str
        The mode of operation. Must be 'replace_text' or 'append_data'.
    replace_with_value : str, optional
        If `replace_mode` is 'replace_text', the string to replace `find_value` with.
        Required if `replace_mode` is 'replace_text'.
    target_column_for_replace : str, optional
        If `replace_mode` is 'replace_text', the column in `df` where the replacement
        should occur. If None, defaults to `column_to_search`.
    lookup_df : pd.DataFrame, optional
        If `replace_mode` is 'append_data', the DataFrame to get data from.
        Required if `replace_mode` is 'append_data'.
    lookup_key_column_lookup : str, optional
        If `replace_mode` is 'append_data', the column in `lookup_df` that contains
        the `find_value` for joining. Required if `replace_mode` is 'append_data'.
    columns_to_append : list of str, optional
        If `replace_mode` is 'append_data', a list of column names from `lookup_df` to append to `df`.
        Required if `replace_mode` is 'append_data'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified operations applied.

    Raises
    ------
    ValueError
        If invalid `search_criteria` or `replace_mode` is provided, or if required
        parameters for a specific mode are missing.

    Examples
    --------
    >>> import pandas as pd
    >>> df_data = {'id': [1, 2, 3, 4], 'text_col': ['apple pie', 'banana split', 'cherry tart', 'date cake']}
    >>> df = pd.DataFrame(df_data)
    >>>
    >>> # Example 1: Replace text
    >>> new_df_replace = find_replace(df, 'text_col', 'pie', 'contains', 'replace_text',
    ...                               replace_with_value='crumble', target_column_for_replace='text_col')
    >>> print(new_df_replace.to_string())
       id     text_col
    0   1  apple crumble
    1   2   banana split
    2   3    cherry tart
    3   4      date cake
    >>>
    >>> # Example 2: Append data
    >>> lookup_data = {'fruit': ['apple', 'banana', 'cherry'], 'color': ['red', 'yellow', 'red']}
    >>> lookup_df = pd.DataFrame(lookup_data)
    >>> new_df_append = find_replace(df, 'text_col', 'apple', 'contains', 'append_data',
    ...                              lookup_df=lookup_df, lookup_key_column_lookup='fruit', columns_to_append=['color'])
    >>> print(new_df_append.to_string())
       id     text_col  color
    0   1    apple pie    red
    1   2   banana split    NaN
    2   3    cherry tart    NaN
    3   4      date cake    NaN
    """
    result_df = df.copy()

    valid_search_criteria = ['contains', 'starts_with', 'equals']
    if search_criteria not in valid_search_criteria:
        raise ValueError(f"Invalid search_criteria. Must be one of {valid_search_criteria}")

    valid_replace_modes = ['replace_text', 'append_data']
    if replace_mode not in valid_replace_modes:
        raise ValueError(f"Invalid replace_mode. Must be one of {valid_replace_modes}")

    # Create a boolean mask for rows that match the find_value in column_to_search
    if search_criteria == 'contains':
        match_mask = result_df[column_to_search].astype(str).str.contains(find_value, na=False)
    elif search_criteria == 'starts_with':
        match_mask = result_df[column_to_search].astype(str).str.startswith(find_value, na=False)
    elif search_criteria == 'equals':
        match_mask = result_df[column_to_search].astype(str) == find_value
    
    if replace_mode == 'replace_text':
        if replace_with_value is None:
            raise ValueError("replace_with_value is required for 'replace_text' mode.")
        
        target_col = target_column_for_replace if target_column_for_replace is not None else column_to_search
        if target_col not in result_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        # Ensure target_col is string type for .str.replace
        result_df[target_col] = result_df[target_col].astype(str)
        
        # Apply replacement only to the matched rows in the target column
        result_df.loc[match_mask, target_col] = result_df.loc[match_mask, target_col].str.replace(find_value, replace_with_value)

    elif replace_mode == 'append_data':
        if lookup_df is None or lookup_key_column_lookup is None or columns_to_append is None:
            raise ValueError("lookup_df, lookup_key_column_lookup, and columns_to_append are required for 'append_data' mode.")
        
        if not isinstance(columns_to_append, list):
            raise ValueError("columns_to_append must be a list of strings.")

        if lookup_key_column_lookup not in lookup_df.columns:
            raise ValueError(f"Lookup key column '{lookup_key_column_lookup}' not found in lookup_df.")
        for col in columns_to_append:
            if col not in lookup_df.columns:
                raise ValueError(f"Column to append '{col}' not found in lookup_df.")

        # If no rows match the find_value condition, return the original df copy
        if not match_mask.any():
            return result_df

        # Select only the necessary columns from lookup_df for merging
        lookup_cols_to_merge = [lookup_key_column_lookup] + columns_to_append
        lookup_df_filtered = lookup_df[lookup_cols_to_merge].drop_duplicates(subset=[lookup_key_column_lookup])

        # Create a temporary column in result_df for joining, only for the matched rows
        temp_join_col = f"_temp_join_key_{np.random.randint(100000, 999999)}"
        result_df[temp_join_col] = np.nan # Initialize with NaN
        result_df.loc[match_mask, temp_join_col] = find_value # Set find_value as key for matched rows

        # Perform a left merge using the temporary join column
        merged_df = pd.merge(
            result_df,
            lookup_df_filtered,
            how='left',
            left_on=temp_join_col,
            right_on=lookup_key_column_lookup,
            suffixes=('', '_lookup')
        )
        
        # Drop the temporary join column and the lookup key column from the merged result
        # The lookup key column from lookup_df is dropped if it's not explicitly requested in columns_to_append
        columns_to_drop = [temp_join_col]
        if lookup_key_column_lookup not in columns_to_append:
            columns_to_drop.append(lookup_key_column_lookup)
        
        merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')
        
        return merged_df
        
    return result_df