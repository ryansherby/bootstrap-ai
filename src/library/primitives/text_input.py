def text_input(df, column_headers, cell_values):
    """
    Allows for manual data entry of a small dataset directly in the workflow.

    This function creates a new pandas DataFrame from provided column headers
    and cell values, simulating manual data entry. The 'df' argument is
    included to satisfy a general constraint but is ignored as this function
    generates a new DataFrame from scratch.

    Parameters
    ----------
    df : pandas.DataFrame
        This argument is ignored. It is included to satisfy a general
        constraint that the first argument must be 'df'.
    column_headers : list of str
        A list of strings representing the column names for the new DataFrame.
    cell_values : list of list
        A list of lists, where each inner list represents a row of data.
        The number of elements in each inner list must match the number
        of `column_headers`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame constructed from the provided headers and values.

    Examples
    --------
    >>> # Assume pd is already imported
    >>> headers = ['Name', 'Age', 'City']
    >>> data = [
    ...     ['Alice', 30, 'New York'],
    ...     ['Bob', 24, 'Los Angeles'],
    ...     ['Charlie', 35, 'Chicago']
    ... ]
    >>> # The 'df' argument is required but ignored for this function's purpose.
    >>> new_df = text_input(None, column_headers=headers, cell_values=data)
    >>> print(new_df)
          Name  Age         City
    0    Alice   30     New York
    1      Bob   24  Los Angeles
    2  Charlie   35      Chicago
    """
    # The 'df' argument is ignored as this function creates a new DataFrame.
    # No need for df.copy() as we are creating a new one.
    new_df = pd.DataFrame(cell_values, columns=column_headers)
    return new_df