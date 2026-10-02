def regex_processor(df, column_to_parse, regex_pattern, output_method, replace_string=None):
    """
    Processes a DataFrame column using regular expressions based on the specified output method.

    This function applies regular expression operations to a designated column
    of a DataFrame. It supports matching, parsing, replacing, or tokenizing
    data based on the provided regular expression pattern. The function
    returns a new DataFrame without modifying the original.

    Parameters
    ----------
    df : pd.DataFrame
        The input pandas DataFrame.
    column_to_parse : str
        The name of the column in `df` to apply the regular expression to.
    regex_pattern : str
        The regular expression pattern to use for processing.
    output_method : str
        The method to apply:
        - 'match': Returns a new column (0 or 1) indicating if the pattern matches.
        - 'parse': Extracts groups from the pattern into new columns.
        - 'replace': Replaces occurrences of the pattern in the column.
        - 'tokenize': Extracts all non-overlapping matches into a list.
    replace_string : str, optional
        The string to use as a replacement when `output_method` is 'replace'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the results of the regex operation.

    Raises
    ------
    ValueError
        If `output_method` is 'replace' but `replace_string` is not provided,
        or if an unsupported `output_method` is specified.
    """
    processed_df = df.copy()
    
    if column_to_parse not in processed_df.columns:
        raise ValueError(f"Column '{column_to_parse}' not found in the DataFrame.")

    if output_method == 'match':
        processed_df[f'{column_to_parse}_match'] = (
            processed_df[column_to_parse]
            .str.contains(regex_pattern, na=False)
            .astype(int)
        )

    elif output_method == 'parse':
        extracted_data = processed_df[column_to_parse].str.extract(regex_pattern)
        if not extracted_data.empty:
            extracted_data.columns = [
                f'{column_to_parse}_parsed_{i}' for i in range(extracted_data.shape[1])
            ]
            processed_df = pd.concat([processed_df, extracted_data], axis=1)

    elif output_method == 'replace':
        if replace_string is None:
            raise ValueError("For 'replace' output_method, 'replace_string' must be provided.")
        processed_df[column_to_parse] = processed_df[column_to_parse].str.replace(
            regex_pattern, replace_string, regex=True
        )

    elif output_method == 'tokenize':
        processed_df[f'{column_to_parse}_tokens'] = (
            processed_df[column_to_parse].str.findall(regex_pattern)
        )

    else:
        raise ValueError(
            f"Unsupported output_method: '{output_method}'. "
            "Choose from 'match', 'parse', 'replace', 'tokenize'."
        )

    return processed_df