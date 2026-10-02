def get_directory_file_metadata(df, directory_path, file_spec="*", include_subdirectories=False):
    """
    Provides metadata for all files within a specific folder, optionally including subdirectories.

    Parameters
    ----------
    df : pd.DataFrame
        Placeholder DataFrame. This function does not modify or use this DataFrame.
    directory_path : str
        The file path to the directory to search.
    file_spec : str, optional
        Wildcard pattern for file names (e.g., "*.xlsx", "report_*.csv").
        Defaults to "*" (all files).
    include_subdirectories : bool, optional
        If True, search includes all subdirectories recursively. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing metadata for the found files, with columns such as
        'file_name', 'full_path', 'size_bytes', 'creation_time', 'modification_time'.

    Examples
    --------
    >>> import os
    >>> import pandas as pd
    >>> # Create a dummy directory and files for demonstration
    >>> if not os.path.exists("temp_dir"):
    >>>     os.makedirs("temp_dir")
    >>> with open("temp_dir/file1.txt", "w") as f:
    >>>     f.write("hello")
    >>> with open("temp_dir/file2.csv", "w") as f:
    >>>     f.write("a,b\n1,2")
    >>> if not os.path.exists("temp_dir/sub_dir"):
    >>>     os.makedirs("temp_dir/sub_dir")
    >>> with open("temp_dir/sub_dir/sub_file.txt", "w") as f:
    >>>     f.write("sub content")
    >>>
    >>> dummy_df = pd.DataFrame() # Input df is not used
    >>> metadata_df = get_directory_file_metadata(dummy_df, "temp_dir", file_spec="*.txt", include_subdirectories=True)
    >>> print(metadata_df[['file_name', 'full_path']].sort_values('file_name'))
    # Expected output:
    #        file_name                  full_path
    # 0      file1.txt          temp_dir/file1.txt
    # 1   sub_file.txt  temp_dir/sub_dir/sub_file.txt
    >>>
    >>> # Clean up dummy directory
    >>> os.remove("temp_dir/file1.txt")
    >>> os.remove("temp_dir/file2.csv")
    >>> os.remove("temp_dir/sub_dir/sub_file.txt")
    >>> os.rmdir("temp_dir/sub_dir")
    >>> os.rmdir("temp_dir")
    """
    file_metadata_list = []

    if include_subdirectories:
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                if fnmatch.fnmatch(file_name, file_spec):
                    full_path = os.path.join(root, file_name)
                    try:
                        stat_info = os.stat(full_path)
                        file_metadata_list.append({
                            "file_name": file_name,
                            "full_path": full_path,
                            "size_bytes": stat_info.st_size,
                            "creation_time": pd.to_datetime(stat_info.st_ctime, unit='s'),
                            "modification_time": pd.to_datetime(stat_info.st_mtime, unit='s')
                        })
                    except OSError:
                        pass # File might be inaccessible or deleted during walk
    else:
        try:
            for file_name in os.listdir(directory_path):
                full_path = os.path.join(directory_path, file_name)
                if os.path.isfile(full_path) and fnmatch.fnmatch(file_name, file_spec):
                    try:
                        stat_info = os.stat(full_path)
                        file_metadata_list.append({
                            "file_name": file_name,
                            "full_path": full_path,
                            "size_bytes": stat_info.st_size,
                            "creation_time": pd.to_datetime(stat_info.st_ctime, unit='s'),
                            "modification_time": pd.to_datetime(stat_info.st_mtime, unit='s')
                        })
                    except OSError:
                        pass # File might be inaccessible
        except OSError:
            pass # Directory does not exist or path is not a directory

    return pd.DataFrame(file_metadata_list)