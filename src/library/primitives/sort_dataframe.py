def sort_dataframe(df, sort_by_columns, sort_orders, use_dictionary_order=False):
    """
    Reorders records in a DataFrame based on one or more specific fields.

    This function sorts the DataFrame by the specified columns and orders.
    It creates a copy of the input DataFrame to ensure immutability.

    Parameters
    ----------
    df : pd.DataFrame
        The input Pandas DataFrame to be sorted.
    sort_by_columns : list of str
        A list of column names to sort the DataFrame by.
    sort_orders : list of bool
        A list of boolean values corresponding to `sort_by_columns`,
        where True means ascending order and False means descending order.
        The length of this list must match `sort_by_columns`.
    use_dictionary_order : bool, optional
        If True, attempts to use language-specific alphanumeric sorting.
        Note: Without external libraries (e.g., 'natsort' or 'locale'),
        this function defaults to standard string comparison for sorting.
        Defaults to False.

    Returns
    -------
    pd.DataFrame
        A new DataFrame sorted according to the specified criteria.

    Raises
    ------
    ValueError
        If the lengths of `sort_by_columns` and `sort_orders` do not match.
        If any column in `sort_by_columns` is not found in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'Name': ['Alice', 'Bob', 'Charlie', 'Alice'],
    ...         'Age': [25, 30, 35, 20],
    ...         'City': ['New York', 'London', 'Paris', 'London']}
    >>> df = pd.DataFrame(data)
    >>> sorted_df = sort_dataframe(df,
    ...                            sort_by_columns=['Name', 'Age'],
    ...                            sort_orders=[True, False])
    """
    if len(sort_by_columns) != len(sort_orders):
        raise ValueError("Lengths of 'sort_by_columns' and 'sort_orders' must match.")
    
    for col in sort_by_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    
    # Create a copy to maintain immutability of the original df
    df_sorted = df.copy()
    
    # Standard Pandas sorting
    df_sorted = df_sorted.sort_values(by=sort_by_columns, ascending=sort_orders)
    
    return df_sorted