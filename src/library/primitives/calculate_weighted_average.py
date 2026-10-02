def calculate_weighted_average(df, value_field, weight_field, grouping_fields=None, output_field_name='weighted_average'):
    """
    Calculates the weighted average of a specified value field, optionally grouped by other fields.

    The function creates a new column in the DataFrame containing the weighted average.
    It does not modify the input DataFrame in-place.

    Parameters
    ----------
    df : pd.DataFrame
        The input pandas DataFrame.
    value_field : str
        The name of the numeric column containing the data to be averaged.
    weight_field : str
        The name of the numeric column containing the weight (importance) of the value.
    grouping_fields : list of str, optional
        A list of column names to group by before calculating the weighted average.
        If None or empty, the weighted average is calculated for the entire DataFrame.
        Defaults to None.
    output_field_name : str, optional
        The name of the new column to store the calculated weighted average.
        Defaults to 'weighted_average'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the additional column containing the weighted average.

    Raises
    ------
    ValueError
        If `value_field` or `weight_field` are not found in the DataFrame,
        or if they are not numeric.
        If any `grouping_fields` are not found in the DataFrame.

    Examples
    --------
    >>> data = {'category': ['A', 'A', 'B', 'B'],
    ...         'value': [10, 20, 30, 40],
    ...         'weight': [1, 2, 1, 3]}
    >>> df = pd.DataFrame(data)
    >>> df_weighted_avg = calculate_weighted_average(df, 'value', 'weight')
    >>> print(df_weighted_avg[['value', 'weight', 'weighted_average']])
       value  weight  weighted_average
    0     10       1         28.571429
    1     20       2         28.571429
    2     30       1         28.571429
    3     40       3         28.571429

    >>> df_grouped_weighted_avg = calculate_weighted_average(df, 'value', 'weight', grouping_fields=['category'])
    >>> print(df_grouped_weighted_avg[['category', 'value', 'weight', 'weighted_average']])
      category  value  weight  weighted_average
    0        A     10       1         16.666667
    1        A     20       2         16.666667
    2        B     30       1         37.500000
    3        B     40       3         37.500000
    """
    df_copy = df.copy()

    # Validate fields
    for field in [value_field, weight_field]:
        if field not in df_copy.columns:
            raise ValueError(f"Field '{field}' not found in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(df_copy[field]):
            raise ValueError(f"Field '{field}' must be numeric.")

    if grouping_fields:
        for field in grouping_fields:
            if field not in df_copy.columns:
                raise ValueError(f"Grouping field '{field}' not found in the DataFrame.")

    # Define the weighted average calculation function for use with transform
    def _calculate_weighted_average_series(group_df):
        # Ensure weights are not zero to avoid division by zero
        total_weight = group_df[weight_field].sum()
        if total_weight == 0:
            # If total weight is zero, return NaN to indicate undefined weighted average
            return np.nan
        return (group_df[value_field] * group_df[weight_field]).sum() / total_weight

    if grouping_fields:
        # Calculate weighted average per group and broadcast back to original DataFrame shape
        df_copy[output_field_name] = df_copy.groupby(grouping_fields).transform(_calculate_weighted_average_series)
    else:
        # Calculate overall weighted average and assign to all rows
        overall_weighted_avg = _calculate_weighted_average_series(df_copy)
        df_copy[output_field_name] = overall_weighted_avg

    return df_copy