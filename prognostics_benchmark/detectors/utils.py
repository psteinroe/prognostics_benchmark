from functools import reduce


def add_rul(df):
    """
    Add remaining useful life as 'rul' to df assuming that the failure occurs at the last time step.
    :param df: input dataframe
    :return: dataframe with 'rul' column added
    """
    df = df.copy()
    max_timestamp = df.index.max()
    df['rul'] = max_timestamp - df.index
    return df


def rul_2_labels(df, lead_time, prediction_horizon):
    df = df.copy()
    df['label'] = df['rul'].apply(lambda x: -1 if x < lead_time else 1 if x < prediction_horizon else 0)
    df.drop('rul', axis=1, inplace=True)
    return df[df.label >= 0]


def align_columns(df, sorted_feature_names):
    """
    aligns columns of given dataframe to order of sorted feature names. Any unknown column is dropped.
    :param df: data frame
    :param sorted_feature_names: sorted list of feature names
    :return: aligned dataframe
    """
    unknown_cols = [column for column in df.columns.tolist() if column not in sorted_feature_names]
    if len(unknown_cols) > 0:
        df.drop(unknown_cols, axis=1, inplace=True)

    missing_features = [feature for feature in sorted_feature_names if feature not in df.columns.tolist()]
    if len(missing_features) > 0:
        df = df.reindex(columns=df.columns.tolist() + missing_features)

    return df.reindex(sorted(df.columns), axis=1)


def dot_to_json(a):
    output = {}
    for key, value in a.items():
        key_split = key.split('.')
        if len(key_split) == 1:
            output[key] = value
            continue
        path = key_split[1:]
        target = reduce(lambda d, k: d.setdefault(k, {}), path[:-1], output)
        target[path[-1]] = value
    return output
