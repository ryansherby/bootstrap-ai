def browse(df):
    """
    Generates a comprehensive data profile for a DataFrame, including data quality metrics
    and statistics for potential visualizations.

    The function calculates null percentages, 'OK' (non-null) percentages, and
    provides data for histograms, top values, and data type distributions.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to profile.

    Returns
    -------
    dict
        A dictionary containing the data profile overview and individual column profiles.
    """
    df_copy = df.copy()
    profile_data = {}

    # --- 1. Overview ---
    profile_data['overview'] = {
        'shape': df_copy.shape,
        'columns': df_copy.columns.tolist(),
        'data_types_distribution': df_copy.dtypes.astype(str).value_counts().to_dict()
    }

    # --- 2. Column Profiles ---
    column_profiles = {}
    for col in df_copy.columns:
        col_profile = {}
        series = df_copy[col]

        # Basic Metadata
        col_profile['dtype'] = str(series.dtype)
        total_rows = len(series)
        null_count = series.isnull().sum()
        not_null_count = series.notnull().sum()

        # Quality Metrics
        col_profile['null_percentage'] = (null_count / total_rows * 100) if total_rows > 0 else 0.0
        col_profile['ok_percentage'] = (not_null_count / total_rows * 100) if total_rows > 0 else 0.0
        col_profile['not_ok_percentage'] = 100.0 - col_profile['ok_percentage']
        col_profile['unique_values'] = series.nunique(dropna=False)

        # Categorical Analysis: Top Values
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            top_n = 5
            top_values = series.value_counts(dropna=False).head(top_n)
            col_profile['top_values'] = [{'value': str(idx), 'count': int(count)} for idx, count in top_values.items()]
        else:
            col_profile['top_values'] = []

        # Numerical Analysis: Stats and Histograms
        if pd.api.types.is_numeric_dtype(series):
            col_profile['min'] = series.min()
            col_profile['max'] = series.max()
            col_profile['mean'] = series.mean()
            col_profile['std'] = series.std()
            col_profile['median'] = series.median()

            # Generate histogram bins using NumPy
            if not series.dropna().empty and series.nunique(dropna=True) > 1:
                counts, bins = np.histogram(series.dropna(), bins='auto')
                col_profile['histogram_data'] = {
                    'bins': bins.tolist(),
                    'counts': counts.tolist()
                }
            else:
                col_profile['histogram_data'] = {'bins': [], 'counts': []}
        else:
            # Non-numeric columns get null placeholders for stats
            stats_keys = ['min', 'max', 'mean', 'std', 'median']
            for key in stats_keys:
                col_profile[key] = None
            col_profile['histogram_data'] = {'bins': [], 'counts': []}

        column_profiles[col] = col_profile

    profile_data['column_profiles'] = column_profiles

    return profile_data