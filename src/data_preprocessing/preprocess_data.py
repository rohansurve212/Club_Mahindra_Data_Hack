# Import Libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def separate_features_and_targets(data_df):
    """
    Separate features from target
    :param data_df: full_data
    :return: a tuple of features and target
    """
    date_cols = ['booking_date', 'checkin_date', 'checkout_date']
    cols_to_exclude = ['reservation_id', 'amount_spent_per_room_night_scaled', 'memberid'] + date_cols
    cols_to_use = [col for col in data_df.columns if col not in cols_to_exclude]
    data_df_X = data_df[cols_to_use]
    data_df_y = data_df['amount_spent_per_room_night_scaled']
    print(data_df_X.shape, data_df_y.shape)

    return data_df_X, data_df_y


def separate_num_cat_features(data_df_x):
    """
    Takes a pandas Dataframe as input and
    separates numerical features from categorical features
    :param data_df_x: full_data
    :return: tuple of numerical features and categorical features
    """
    data_df_num_feature_names = data_df_x.select_dtypes(include=[np.number]).columns.tolist()
    data_df_cat_feature_names = data_df_x.select_dtypes(include=['object']).columns.tolist()

    return data_df_num_feature_names, data_df_cat_feature_names


def transform_num_features(df_features, df_num_feature_columns):
    """
            Takes a pandas Dataframe (part of the black_friday_data_hack project) as input and
            transforms the numerical features (imputing missing values with median value and
            normalizing the values in each column such that minimum is 0 and maximum is 1)
            Arguments: data - DataFrame
            Returns: A numpy array of transformed numerical features
    """

    # Let's build a pipeline to transform numerical features
    df_num = df_features[df_num_feature_columns]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('normalizer', MinMaxScaler())
    ])

    df_num_tr = num_pipeline.fit_transform(df_num)

    return df_num_tr


def transform_cat_features(df_features, df_cat_feature_columns):
    """
            Takes a pandas DataFrame (part of the black_friday_data_hack project) as input and
            transforms the categorical features (imputing missing values with most frequent occurence
            value and performing one-hot encoding)
            Arguments: data - DataFrame
            Returns: A numpy array of transformed categorical features
    """

    # Let's build a pipeline to transform categorical features
    df_cat = df_features[df_cat_feature_columns]

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder())
    ])

    df_cat_tr = cat_pipeline.fit_transform(df_cat)

    return df_cat_tr


def transform_all_features(data_df_x, data_df_num_feature_names, data_df_cat_feature_names):
    """
    transforms all the features to be ready to be fed to machine learning models
    :param data_df_x: data features
    :param data_df_num_feature_names: list of names of numerical features
    :param data_df_cat_feature_names: list of names of categorical features
    :return: A numpy array of transformed features
    """

    # Let's create the full pipeline
    try:
        from sklearn.compose import ColumnTransformer
    except ImportError:
        from future_encoders import ColumnTransformer

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('normalizer', MinMaxScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder())
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, data_df_num_feature_names),
        ('cat', cat_pipeline, data_df_cat_feature_names)
    ])

    data_df_processed = full_pipeline.fit_transform(data_df_x)

    return data_df_processed
