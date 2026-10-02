def output_data(df, output_file_path_or_dsn, output_options='overwrite', sheet_name=None, pre_sql=None, post_sql=None, table_name=None):
    """
    Writes a DataFrame to a specified file (CSV, XLSX) or a database table.

    This function ensures immutability by operating on a copy of the input DataFrame.
    It supports various output options like overwriting, appending, or creating new files/tables.
    For database operations, it can execute SQL statements before and after the data write.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be written.
    output_file_path_or_dsn : str
        The path to the output file (e.g., 'data.csv', 'report.xlsx') or a database
        connection string (DSN) (e.g., 'postgresql://user:pass@host:port/db').
    output_options : str, optional
        Specifies how to handle existing data.
        - 'overwrite': Overwrites the file/table if it exists. For files, this means
                       creating a new file. For databases, it drops and recreates the table.
        - 'append': Appends data to an existing file/table.
        - 'create_new': (Database only) Creates a new table. Raises an error if it already exists.
                        For files, this option behaves like 'overwrite' due to limitations
                        without external file system checks.
        The default is 'overwrite'.
    sheet_name : str, optional
        Required when writing to an Excel file. Specifies the name of the sheet to write to.
        If None for Excel, defaults to 'Sheet1'.
    pre_sql : str or list of str, optional
        SQL statement(s) to execute before writing data to a database.
        Only applicable when `output_file_path_or_dsn` is a DSN.
    post_sql : str or list of str, optional
        SQL statement(s) to execute after writing data to a database.
        Only applicable when `output_file_path_or_dsn` is a DSN.
    table_name : str, optional
        Required when writing to a database. Specifies the name of the table to write to.

    Raises
    ------
    ValueError
        If `table_name` is not provided for database writes, or if an unsupported
        file extension is used, or if 'create_new' option is used for a database
        and the table already exists.
    Exception
        For other errors during file or database operations.

    Returns
    -------
    None

    Examples
    --------
    >>> import pandas as pd
    >>> # Assume pd and sqlalchemy are already imported
    >>> data = {'col1': [1, 2], 'col2': ['A', 'B']}
    >>> df_input = pd.DataFrame(data)

    >>> # Example 1: Write to a CSV file (overwrite)
    >>> # output_data(df_input, 'output_data.csv', output_options='overwrite')
    >>> # pd.read_csv('output_data.csv')
    >>> #    col1 col2
    >>> # 0     1    A
    >>> # 1     2    B

    >>> # Example 2: Write to an Excel file (append to a sheet)
    >>> # output_data(df_input, 'output_report.xlsx', output_options='append', sheet_name='MyData')
    >>> # pd.read_excel('output_report.xlsx', sheet_name='MyData')
    >>> #    col1 col2
    >>> # 0     1    A
    >>> # 1     2    B

    >>> # Example 3: Write to a PostgreSQL database table (overwrite)
    >>> # Assume 'postgresql://user:pass@localhost:5432/mydb' is a valid DSN
    >>> # output_data(df_input, 'postgresql://user:pass@localhost:5432/mydb',
    >>> #             output_options='overwrite', table_name='my_output_table',
    >>> #             pre_sql='DROP TABLE IF EXISTS my_temp_table;',
    >>> #             post_sql='ANALYZE my_output_table;')
    >>> # # To verify (requires a database connection):
    >>> # # from sqlalchemy import create_engine
    >>> # # engine = create_engine('postgresql://user:pass@localhost:5432/mydb')
    >>> # # pd.read_sql('SELECT * FROM my_output_table', engine)
    >>> # #    col1 col2
    >>> # # 0     1    A
    >>> # # 1     2    B
    """
    df_copy = df.copy()

    # Determine if it's a file or database operation based on extension
    is_file_output = False
    if isinstance(output_file_path_or_dsn, str):
        output_file_path_or_dsn_lower = output_file_path_or_dsn.lower()
        if output_file_path_or_dsn_lower.endswith(('.csv', '.xlsx')):
            is_file_output = True

    if is_file_output:
        file_path = output_file_path_or_dsn
        file_extension = file_path.split('.')[-1].lower()

        if output_options == 'create_new':
            # For files, 'create_new' behaves like 'overwrite' due to no 'os' import allowed
            # to check for file existence. Documented in docstring.
            output_options = 'overwrite'

        if file_extension == 'csv':
            mode = 'w' if output_options == 'overwrite' else 'a'
            header = True if mode == 'w' else False # Append mode should not write header again
            df_copy.to_csv(file_path, mode=mode, header=header, index=False)
        elif file_extension == 'xlsx':
            if sheet_name is None:
                sheet_name = 'Sheet1' # Default sheet name for Excel

            if output_options == 'overwrite':
                # Overwrites the entire Excel file if it exists, or creates new.
                # Writes to the specified sheet.
                df_copy.to_excel(file_path, sheet_name=sheet_name, index=False)
            elif output_options == 'append':
                # Appends to the Excel file. If sheet_name exists, appends to it.
                # If sheet_name does not exist, creates a new sheet.
                # This requires 'openpyxl' to be installed, but not explicitly imported.
                df_copy.to_excel(file_path, sheet_name=sheet_name, mode='a', header=False, index=False)
            else:
                raise ValueError(f"Invalid output_options '{output_options}' for Excel file. "
                                 "Use 'overwrite' or 'append'.")
        else:
            raise ValueError(f"Unsupported file extension: .{file_extension}. Only .csv and .xlsx are supported.")
    else: # Database output
        if table_name is None:
            raise ValueError("For database output, 'table_name' must be provided.")

        # SQLAlchemy engine creation
        engine = sqlalchemy.create_engine(output_file_path_or_dsn)

        # Handle pre_sql
        if pre_sql:
            with engine.connect() as connection:
                if isinstance(pre_sql, str):
                    connection.execute(sqlalchemy.text(pre_sql))
                elif isinstance(pre_sql, list):
                    for stmt in pre_sql:
                        connection.execute(sqlalchemy.text(stmt))
                connection.commit() # Commit after pre_sql statements

        # Determine if_exists for to_sql
        if_exists_option = 'fail' # Default for 'create_new'
        if output_options == 'overwrite':
            if_exists_option = 'replace'
        elif output_options == 'append':
            if_exists_option = 'append'
        elif output_options == 'create_new':
            if_exists_option = 'fail' # Explicitly fail if table exists

        try:
            df_copy.to_sql(name=table_name, con=engine, if_exists=if_exists_option, index=False)
        except Exception as e:
            # Re-raise all exceptions from to_sql
            raise # Re-raise all exceptions from to_sql

        # Handle post_sql
        if post_sql:
            with engine.connect() as connection:
                if isinstance(post_sql, str):
                    connection.execute(sqlalchemy.text(post_sql))
                elif isinstance(post_sql, list):
                    for stmt in post_sql:
                        connection.execute(sqlalchemy.text(stmt))
                connection.commit() # Commit after post_sql statements
