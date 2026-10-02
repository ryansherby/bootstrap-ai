def formula(df, output_column, expression, data_type):
    """
    Creates or updates a column in a DataFrame using a specified expression.

    The function evaluates a given expression string against the DataFrame's columns
    and assigns the result to a new or existing column. The resulting column
    can then be cast to a specified data type.

    Parameters
    ----------
    df : pd.DataFrame
        The input pandas DataFrame.
    output_column : str
        The name of the column to create or update.
    expression : str
        A string representing the formula to apply. This expression can reference
        existing column names directly (e.g., "col_a + col_b") and can use
        numpy functions (e.g., "np.log(col_c)").
    data_type : str
        The desired data type for the output column (e.g., "int64", "float64", "string").

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified column created or updated.

    Examples
    --------
    >>> import pandas as pd
    >>> df_input = pd.DataFrame({'A': [1, 3], 'B': [2, 4]})
    >>> df_output = formula(df_input, 'C', 'A + B', 'int64')
    >>> print(df_output)
       A  B  C
    0  1  2  3
    1  3  4  7
    """
    df_copy = df.copy()
    
    df_copy[output_column] = df_copy.eval(expression)
    
    df_copy[output_column] = df_copy[output_column].astype(data_type)
    
    return df_copy