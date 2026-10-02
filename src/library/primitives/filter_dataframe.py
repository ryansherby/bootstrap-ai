def filter_dataframe(df, column=None, operator=None, value=None, expression=None):
    """
    Splits data into two streams (True/False) based on a condition.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column : str, optional
        The column to apply the basic filter on. Required for basic filtering.
    operator : str, optional
        The operator for basic filtering (e.g., 'equals', 'not_equals', 'is_null',
        'not_null', 'greater_than', 'less_than', 'greater_than_or_equal',
        'less_than_or_equal', 'isin'). Required for basic filtering.
    value : any, optional
        The value to compare against for basic filtering. Required for basic filtering,
        unless operator is 'is_null' or 'not_null'.
    expression : str, optional
        A custom filter expression (e.g., "[Age] > 18 and [Status] == 'Active'").
        If provided, basic filter parameters (column, operator, value) are ignored.

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing two DataFrames: (df_true, df_false).
        df_true contains rows where the condition is met.
        df_false contains rows where the condition is not met.

    Raises
    ------
    ValueError
        If neither basic filter parameters nor an expression are provided,
        or if an invalid operator is specified.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    ...         'Age': [25, 30, 35, 25],
    ...         'City': ['NY', 'LA', 'NY', 'SF'],
    ...         'Score': [85, 90, None, 78]}
    >>> df = pd.DataFrame(data)

    >>> # Basic filter example
    >>> df_true_basic, df_false_basic = filter_dataframe(df, column='Age', operator='greater_than', value=28)
    >>> print("True Stream (Basic Filter):")
    >>> print(df_true_basic)
    >>> print("False Stream (Basic Filter):")
    >>> print(df_false_basic)
    True Stream (Basic Filter):
          Name  Age City  Score
    1      Bob   30   LA   90.0
    2  Charlie   35   NY    NaN
    False Stream (Basic Filter):
          Name  Age City  Score
    0    Alice   25   NY   85.0
    3    David   25   SF   78.0

    >>> # Custom filter example
    >>> df_true_custom, df_false_custom = filter_dataframe(df, expression="Age > 25 and City == 'NY'")
    >>> print("True Stream (Custom Filter):")
    >>> print(df_true_custom)
    >>> print("False Stream (Custom Filter):")
    >>> print(df_false_custom)
    True Stream (Custom Filter):
          Name  Age City  Score
    2  Charlie   35   NY    NaN
    False Stream (Custom Filter):
          Name  Age City  Score
    0    Alice   25   NY   85.0
    1      Bob   30   LA   90.0
    3    David   25   SF   78.0
    """
    df_copy = df.copy()
    condition = None

    if expression:
        df_true = df_copy.query(expression)
        df_false = df_copy.drop(df_true.index)
    elif column is not None and operator is not None:
        # Basic filtering
        if operator == 'equals':
            condition = (df_copy[column] == value)
        elif operator == 'not_equals':
            condition = (df_copy[column] != value)
        elif operator == 'is_null':
            condition = df_copy[column].isnull()
        elif operator == 'not_null':
            condition = df_copy[column].notnull()
        elif operator == 'greater_than':
            condition = (df_copy[column] > value)
        elif operator == 'less_than':
            condition = (df_copy[column] < value)
        elif operator == 'greater_than_or_equal':
            condition = (df_copy[column] >= value)
        elif operator == 'less_than_or_equal':
            condition = (df_copy[column] <= value)
        elif operator == 'isin':
            if not isinstance(value, (list, tuple, set, np.ndarray)):
                raise ValueError("Value for 'isin' operator must be a list, tuple, set, or numpy array.")
            condition = df_copy[column].isin(value)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        df_true = df_copy[condition]
        df_false = df_copy[~condition]
    else:
        raise ValueError("Either 'expression' or 'column', 'operator', and 'value' must be provided.")

    return df_true, df_false