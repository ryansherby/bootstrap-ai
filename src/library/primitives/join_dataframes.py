def join_dataframes(df, right_df, join_mode="fields", left_on=None, right_on=None, how="inner"):
    """
    Combines two pandas DataFrames horizontally based on common fields or record position.

    Parameters
    ----------
    df : pd.DataFrame
        The left DataFrame.
    right_df : pd.DataFrame
        The right DataFrame to join with.
    join_mode : str, optional
        The mode of joining: 'fields' (key-based) or 'position' (index-based).
        Defaults to 'fields'.
    left_on : str or list of str, optional
        Column(s) in `df` to join on. Required if `join_mode` is 'fields'.
    right_on : str or list of str, optional
        Column(s) in `right_df` to join on. Required if `join_mode` is 'fields'.
    how : str, optional
        Type of merge: 'left', 'right', 'outer', 'inner', 'cross'.
        Defaults to 'inner'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame resulting from the join operation.

    Raises
    ------
    ValueError
        If required join keys are missing for 'fields' mode or if an unsupported 
        mode is provided.
    """
    if join_mode == "fields":
        if left_on is None or right_on is None:
            raise ValueError("For 'fields' join_mode, 'left_on' and 'right_on' must be provided.")
        
        return pd.merge(df, right_df, left_on=left_on, right_on=right_on, how=how)

    elif join_mode == "position":
        # Resetting index ensures row-by-row alignment based on integer position
        # rather than existing index labels.
        return pd.concat(
            [df.reset_index(drop=True), right_df.reset_index(drop=True)], 
            axis=1
        )

    else:
        raise ValueError(
            f"Unsupported join_mode: {join_mode}. Expected 'fields' or 'position'."
        )