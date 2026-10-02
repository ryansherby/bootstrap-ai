def convert_datetime_format(df, column_name, convert_mode, date_format=None):
    """
    Converts a column between string and datetime formats, suitable for Alteryx.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column_name : str
        The name of the column to convert.
    convert_mode : str
        The conversion mode. Must be either "String to Date/Time" or "Date/Time to String".
    date_format : str, optional
        The format string to use for conversion.
        - If `convert_mode` is "String to Date/Time", this specifies the input format.
          If None, pandas will attempt to infer the format.
        - If `convert_mode` is "Date/Time to String", this specifies the output format.
          This parameter is required for "Date/Time to String" mode.
          Example formats: '%Y-%m-%d', '%Y/%m/%d %H:%M:%S'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified column converted.

    Raises
    ------
    ValueError
        If an invalid `convert_mode` is provided or `date_format` is missing
        for "Date/Time to String" mode.

    Examples
    --------
    >>> import pandas as pd
    >>> df_input = pd.DataFrame({
    ...     'id': [1, 2],
    ...     'date_col': ['2023-01-15', '2024-02-29']
    ... })
    >>> # Convert string to datetime
    >>> df_converted_dt = convert_datetime_format(df_input, 'date_col', 'String to Date/Time')
    >>> print(df_converted_dt['date_col'].dtype)
    datetime64[ns]
    >>> # Convert datetime to string with a specific format
    >>> df_converted_str = convert_datetime_format(df_converted_dt, 'date_col', 'Date/Time to String', '%Y/%m/%d')
    >>> print(df_converted_str['date_col'].iloc[0])
    2023/01/15
    """
    df_copy = df.copy()

    if convert_mode == "String to Date/Time":
        df_copy[column_name] = pd.to_datetime(df_copy[column_name], format=date_format, errors='coerce')
    elif convert_mode == "Date/Time to String":
        if date_format is None:
            raise ValueError("date_format must be provided when convert_mode is 'Date/Time to String'.")
        # Ensure the column is datetime before converting to string
        if not pd.api.types.is_datetime64_any_dtype(df_copy[column_name]):
             df_copy[column_name] = pd.to_datetime(df_copy[column_name], errors='coerce')
        df_copy[column_name] = df_copy[column_name].dt.strftime(date_format)
    else:
        raise ValueError("Invalid convert_mode. Must be 'String to Date/Time' or 'Date/Time to String'.")

    return df_copy