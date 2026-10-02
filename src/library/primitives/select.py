def select(df, column_configurations, prefix=None, suffix=None):
    """
    Manages columns in a DataFrame by renaming, reordering, and changing data types.

    This function processes a DataFrame based on a list of column configurations,
    allowing for selective inclusion, renaming, type conversion, and optional
    prefix/suffix addition to column names. It ensures the input DataFrame
    remains unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The input pandas DataFrame.
    column_configurations : list of dict
        A list where each dictionary specifies the desired state for a column.
        Each dictionary can contain:
        - 'original_name' (str, required): The current name of the column in `df`.
        - 'output_name' (str, optional): The new name for the column.
        - 'data_type' (str, optional): The target data type for the column.
    prefix : str, optional
        A string to prepend to all output column names.
    suffix : str, optional
        A string to append to all output column names.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with columns processed according to the configurations.
    """
    df_processed = df.copy()

    rename_map = {}
    type_conversion_map = {}
    final_column_order = []

    for config in column_configurations:
        original_col = config.get('original_name')
        
        # Skip if original name is missing or not in the source DataFrame
        if original_col is None or original_col not in df.columns:
            continue

        # Determine the target name
        target_col_name = config.get('output_name', original_col)

        if prefix is not None:
            target_col_name = prefix + target_col_name
        if suffix is not None:
            target_col_name = target_col_name + suffix

        rename_map[original_col] = target_col_name
        
        # Maintain the order specified in the configuration list
        if target_col_name not in final_column_order:
            final_column_order.append(target_col_name)

        # Store data type preference
        data_type = config.get('data_type')
        if data_type is not None:
            type_conversion_map[target_col_name] = data_type

    # Apply renaming
    df_processed = df_processed.rename(columns=rename_map)

    # Reorder and filter columns to match the config order
    existing_cols = [col for col in final_column_order if col in df_processed.columns]
    df_processed = df_processed[existing_cols]

    # Apply type conversions
    for col_name, target_type in type_conversion_map.items():
        if col_name in df_processed.columns:
            try:
                df_processed[col_name] = df_processed[col_name].astype(target_type)
            except (ValueError, TypeError):
                # Silently skip if conversion fails to maintain robustness
                pass

    return df_processed