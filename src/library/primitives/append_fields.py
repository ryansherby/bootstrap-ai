def append_fields(df: pd.DataFrame, df_source: pd.DataFrame, warn_error_setting: str = "allow_all") -> pd.DataFrame:
    """
    Performs a Cartesian join (appends every record of Source to every record of Target).

    This function takes a target DataFrame and a source DataFrame and performs a 
    Cartesian join, effectively appending every record of the source to every 
    record of the target.

    Parameters
    ----------
    df : pd.DataFrame
        The target DataFrame.
    df_source : pd.DataFrame
        The source DataFrame to be appended.
    warn_error_setting : str, optional
        Controls behavior for large results: 'allow_all', 'warn_gt_16', or 'error_gt_16'.
        Defaults to 'allow_all'.

    Returns
    -------
    pd.DataFrame
        The DataFrame resulting from the Cartesian join.

    Raises
    ------
    ValueError
        If `warn_error_setting` is 'error_gt_16' and the result has > 16 records.
    """
    df_target_copy = df.copy()
    df_source_copy = df_source.copy()

    # Create a dummy key to facilitate the Cartesian product (Cross Join)
    df_target_copy['_dummy_key'] = 1
    df_source_copy['_dummy_key'] = 1

    merged_df = pd.merge(df_target_copy, df_source_copy, on='_dummy_key')
    num_records = len(merged_df)

    # Threshold Check logic
    if warn_error_setting == "warn_gt_16" and num_records > 16:
        warnings.warn("Cartesian join resulted in more than 16 records.")
    elif warn_error_setting == "error_gt_16" and num_records > 16:
        raise ValueError(f"Error: Result size {num_records} exceeds the 16-record limit.")

    # Clean up the helper key
    merged_df = merged_df.drop(columns=['_dummy_key'])
    
    return merged_df