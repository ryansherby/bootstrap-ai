def summarize(df, group_by_cols, agg_definitions, separator=', '):
    """
    Aggregates data based on specified grouping columns and aggregation definitions.

    This function performs operations similar to SQL's GROUP BY, allowing
    for various aggregations like sum, count, max, min, mean, unique count,
    concatenation, and finding the longest string.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be aggregated.
    group_by_cols : list of str
        A list of column names to group the DataFrame by. If an empty list,
        the aggregation will be applied to the entire DataFrame.
    agg_definitions : list of dict
        A list of dictionaries, where each dictionary defines one aggregation.
        Each dictionary must contain:
        - 'new_col_name' (str): The name of the new column for the aggregated result.
        - 'original_col' (str): The name of the original column to aggregate.
        - 'method' (str): The aggregation method. Supported methods include:
          'sum', 'count', 'max', 'min', 'mean', 'first', 'last', 'nunique',
          'concatenate', 'longest'.
        For 'concatenate' method, an optional 'separator' key can be included
        in the dictionary to override the default separator for that specific aggregation.
    separator : str, optional
        The default separator to use for 'concatenate' method if not specified
        in the individual aggregation definition. Defaults to ', '.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the aggregated results.

    Raises
    ------
    ValueError
        If an unsupported aggregation method is specified.

    Examples
    --------
    >>> data = {
    ...     'category': ['A', 'A', 'B', 'B', 'A'],
    ...     'value': [1, 2, 3, 4, 5],
    ...     'item': ['X', 'Y', 'Z', 'W', 'V']
    ... }
    >>> df = pd.DataFrame(data)
    >>> group_by_cols = ['category']
    >>> agg_definitions = [
    ...     {'new_col_name': 'total_value', 'original_col': 'value', 'method': 'sum'},
    ...     {'new_col_name': 'item_count', 'original_col': 'item', 'method': 'count'},
    ...     {'new_col_name': 'unique_items', 'original_col': 'item', 'method': 'nunique'},
    ...     {'new_col_name': 'concatenated_items', 'original_col': 'item', 'method': 'concatenate', 'separator': '-'},
    ...     {'new_col_name': 'longest_item', 'original_col': 'item', 'method': 'longest'}
    ... ]
    >>> summarized_df = summarize(df, group_by_cols, agg_definitions)
    >>> print(summarized_df)
      category  total_value  item_count  unique_items concatenated_items longest_item
    0        A            8           3             3              X-Y-V            X
    1        B            7           2             2                Z-W            Z

    >>> # Example without grouping (aggregation on entire DataFrame)
    >>> agg_definitions_no_group = [
    ...     {'new_col_name': 'overall_sum', 'original_col': 'value', 'method': 'sum'},
    ...     {'new_col_name': 'all_items', 'original_col': 'item', 'method': 'concatenate', 'separator': '|'}
    ... ]
    >>> summarized_no_group_df = summarize(df, [], agg_definitions_no_group)
    >>> print(summarized_no_group_df)
       overall_sum all_items
    0           15   X|Y|Z|W|V
    """
    working_df = df.copy()
    
    # Handle no grouping case by adding a dummy column
    if not group_by_cols:
        working_df['_dummy_group_col'] = 0
        current_group_by_cols = ['_dummy_group_col']
    else:
        current_group_by_cols = group_by_cols
        
    grouped = working_df.groupby(current_group_by_cols)
    
    pandas_agg_mapping = {} # Format: {'new_col_name': ('original_col', 'method')}
    custom_agg_series = {} # Format: {'new_col_name': pd.Series}

    for agg_def in agg_definitions:
        new_col = agg_def['new_col_name']
        original_col = agg_def['original_col']
        method = agg_def['method']
        
        if method in ['sum', 'count', 'max', 'min', 'mean', 'first', 'last', 'nunique']:
            pandas_agg_mapping[new_col] = (original_col, method)
        elif method == 'concatenate':
            current_separator = agg_def.get('separator', separator)
            custom_agg_series[new_col] = grouped[original_col].apply(
                lambda x: current_separator.join(x.astype(str).dropna().unique())
            )
        elif method == 'longest':
            custom_agg_series[new_col] = grouped[original_col].apply(
                lambda x: max(x.astype(str), key=len) if x.astype(str).any() else np.nan
            )
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    # Perform pandas aggregations
    if pandas_agg_mapping:
        aggregated_df = grouped.agg(**pandas_agg_mapping)
    else:
        # If only custom aggregations, create a base DataFrame with group keys as index
        if current_group_by_cols:
            # Create a DataFrame from the group keys, ensuring correct index structure
            aggregated_df = grouped.apply(lambda x: x.name).to_frame(name='_temp_col')
            aggregated_df = aggregated_df.drop(columns=['_temp_col'])
        else:
            # This case should be covered by _dummy_group_col, but as a fallback
            aggregated_df = pd.DataFrame(index=[0]) # Single row, single index
            
    # Merge custom aggregations
    for col_name, series in custom_agg_series.items():
        # The .apply() on grouped object already returns a Series with group keys as index.
        # So, direct assignment should work if aggregated_df has the group_by_cols as index.
        aggregated_df[col_name] = series
        
    # Reset index to make group_by_cols regular columns
    result_df = aggregated_df.reset_index()

    # Clean up dummy column if it was used
    if '_dummy_group_col' in result_df.columns:
        result_df = result_df.drop(columns=['_dummy_group_col'])
        
    return result_df