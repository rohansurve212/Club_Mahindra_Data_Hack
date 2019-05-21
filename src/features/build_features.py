# Import libraries
import pandas as pd
import datetime as dt


def date_related_features(data_df):
    """
    For each date column create three separate columns that store the corresponding month, year and week
    For each date column create a column that stores the date in seconds from the reference date (1st January, 1970)
    Create a column that stores the number of days stayed against each observation
    Create a column that stores the number of days of advance booking for each observation
    :param data_df: training set dataframe
    :return: modified training and test set
    """

    date_cols = ['booking_date', 'checkin_date', 'checkout_date']

    for date_col in date_cols:

        data_df[date_col] = pd.to_datetime(data_df[date_col], format='%d/%m/%y')

        data_df[date_col + '_in_seconds'] = (data_df[date_col] - dt.datetime(1970, 1, 1)).dt.total_seconds()

        data_df[date_col + '_month'] = data_df[date_col].dt.month

        data_df[date_col + '_year'] = data_df[date_col].dt.year

        data_df[date_col + '_week'] = data_df[date_col].dt.week

    data_df['days_stay'] = (data_df['checkout_date'] - data_df['checkin_date']).dt.days

    data_df['days_advance_booking'] = (data_df['checkin_date'] - data_df['booking_date']).dt.days

    return data_df

# def merge_train_test(train, test):
#     """
#     Merge the train and test set in order to build features
#     :param train: training set dataframe
#     :param test: test set dataframe
#     :return: a single merged dataframe
#     """
#
#     train = train.drop('amount_spent_per_room_night_scaled', axis=1)
#     full_data = pd.concat([train, test]).reset_index(drop=True)
#     full_data = full_data.sort_values(by='checkin_date').reset_index(drop=True)
#
#     return full_data


def aggregate_features_by_memberid(data_df):
    """
    Get the aggregate values of different columns grouped by memberid
    ['booking_date_in_seconds', 'checkin_date_in_seconds', 'days_stay',
    'days_advance_booking', 'roomnights']: mean of all values
    ['days_stay', 'roomnights']: sum of all values
    ['resort_id']: number of unique values
    :param data_df: full_data
    :return: mini_data
    """
    initial_columns = ['memberid', 'resort_id', 'state_code_residence', 'checkin_date', 'booking_date']

    mini_data_df = data_df[['reservation_id'] + initial_columns]

    # get the mean of important columns
    for col in ['booking_date_in_seconds', 'checkin_date_in_seconds', 'days_stay', 'days_advance_booking',
                'roomnights']:
        if col == 'roomnights':
            data_df[col] = data_df[col].astype('float64')
        gdf = data_df.groupby('memberid')[col].mean().reset_index()
        gdf.columns = ['memberid', 'member_' + col + '_mean']
        mini_data_df = pd.merge(mini_data_df, gdf, on='memberid', how='left')

    # get the sum of important columns
    for col in ['days_stay', 'roomnights']:
        gdf = data_df.groupby('memberid')[col].sum().reset_index()
        gdf.columns = ['memberid', 'member_' + col + '_sum']
        mini_data_df = pd.merge(mini_data_df, gdf, on='memberid', how='left')

    # get the total #unique values of important columns
    for col in ['resort_id']:
        gdf = data_df.groupby('memberid')[col].nunique().reset_index()
        gdf.columns = ['memberid', 'member_' + col + '_nunique']
        mini_data_df = pd.merge(mini_data_df, gdf, on='memberid', how='left')

    return mini_data_df


def cumulative_features_by_memberid(data_df, mini_data_df):
    """
    Take cumulative values of different columns grouped by each memberid and add it to mini_data
    :param data_df: full_data
    :param mini_data_df: mini_data
    :return: modified mini_data
    """
    # Cumulative count of member bookings
    mini_data_df['cumcount_member_bookings'] = data_df.groupby('memberid')['booking_date_in_seconds'].cumcount().values

    # Cumulative sum of member stays at CM resorts (all inclusive)
    mini_data_df['cumsum_member_days_stay'] = data_df.groupby('memberid')['days_stay'].cumsum().values

    # Cumulative sum of total # people travelled with member at CM resorts (all inclusive)
    data_df['total_pax'] = data_df['total_pax'].astype('int64')
    mini_data_df['cumsum_member_total_pax'] = data_df.groupby('memberid')['total_pax'].cumsum().values

    return mini_data_df


def time_gap_shift_1_features(data_df, mini_data_df):
    """
    For each member, compute:
    1. time gap between current booking date and previous booking date/next booking date
    2. time gap between current checkin date and previous checkin date/next checkin date
    3. time gap between current checkout date and previous checkout date/next checkout date
    For each member-resort combination, compute:
    1. time gap between current booking date and previous booking date/next booking date
    2. time gap between current checkin date and previous checkin date/next checkin date
    :param data_df: full_data
    :param mini_data_df: mini_data
    :return: modified mini_data
    """
    # Compute time gap between current booking date and previous booking date/next booking date for each memberid and
    # add it to mini_data
    data_df['prev_booking_date_in_seconds'] = data_df.groupby('memberid')['booking_date_in_seconds'].shift(1)
    mini_data_df['time_gap_booking_prev'] = data_df['booking_date_in_seconds'] - data_df[
        'prev_booking_date_in_seconds']

    data_df['next_booking_date_in_seconds'] = data_df.groupby('memberid')['booking_date_in_seconds'].shift(-1)
    mini_data_df['time_gap_booking_next'] = data_df['next_booking_date_in_seconds'] - data_df[
        'booking_date_in_seconds']

    # Compute time gap between current checkin date and previous checkin date/next checkin date for each memberid and
    # add it to mini_data
    data_df['prev_checkin_date_in_seconds'] = data_df.groupby('memberid')['checkin_date_in_seconds'].shift(1)
    mini_data_df['time_gap_checkin_prev'] = data_df['checkin_date_in_seconds'] - data_df[
        'prev_checkin_date_in_seconds']

    data_df['next_checkin_date_in_seconds'] = data_df.groupby('memberid')['checkin_date_in_seconds'].shift(-1)
    mini_data_df['time_gap_checkin_next'] = data_df['next_checkin_date_in_seconds'] - data_df[
        'checkin_date_in_seconds']

    # Compute time gap between current checkout date and previous checkout date/next checkout date for each memberid
    # and add it to mini_data
    data_df['prev_checkout_date_in_seconds'] = data_df.groupby('memberid')['checkout_date_in_seconds'].shift(1)
    mini_data_df['time_gap_checkout_prev'] = data_df['checkout_date_in_seconds'] - data_df[
        'prev_checkout_date_in_seconds']

    data_df['next_checkout_date_in_seconds'] = data_df.groupby('memberid')['checkout_date_in_seconds'].shift(-1)
    mini_data_df['time_gap_checkout_next'] = data_df['next_checkout_date_in_seconds'] - data_df[
        'checkout_date_in_seconds']

    # Compute time gap between current booking date and previous booking date/next booking date for each
    # memberid-resort_id combination and add it to mini_data
    data_df['prev_resort_booking_date_in_seconds'] = data_df.groupby(['memberid', 'resort_id'])[
        'booking_date_in_seconds'].shift(1)
    mini_data_df['time_gap_booking_prev_resort'] = data_df['booking_date_in_seconds'] - data_df[
        'prev_resort_booking_date_in_seconds']

    data_df['next_resort_booking_date_in_seconds'] = data_df.groupby(['memberid', 'resort_id'])[
        'booking_date_in_seconds'].shift(-1)
    mini_data_df['time_gap_booking_next_resort'] = data_df['next_resort_booking_date_in_seconds'] - data_df[
        'booking_date_in_seconds']

    # Compute time gap between current checkin date and previous checkin date/next checkin date for each
    # memberid-resort_id combination and add it to mini_data
    data_df['prev_resort_checkin_date_in_seconds'] = data_df.groupby(['memberid', 'resort_id'])[
        'checkin_date_in_seconds'].shift(1)
    mini_data_df['time_gap_checkin_prev_resort'] = data_df['checkin_date_in_seconds'] - data_df[
        'prev_resort_checkin_date_in_seconds']

    data_df['next_resort_checkin_date_in_seconds'] = data_df.groupby(['memberid', 'resort_id'])[
        'checkin_date_in_seconds'].shift(-1)
    mini_data_df['time_gap_checkin_next_resort'] = data_df['next_resort_checkin_date_in_seconds'] - data_df[
        'checkin_date_in_seconds']

    return mini_data_df


def time_gap_shift_2_features(data_df, mini_data_df):
    """
    For each member, compute:
    1. time gap between current booking date and second last booking date/next-to-next booking date
    2. time gap between current checkin date and second last checkin date/next-to-next checkin date
    :param data_df: full_data
    :param mini_data_df: mini_data
    :return: modified mini_data
    """
    # Compute time gap between current booking date and second last booking date/next-to-next booking date (shift 2)
    # for each memberid and add it to mini_data
    data_df['prev2_booking_date_in_seconds'] = data_df.groupby('memberid')['booking_date_in_seconds'].shift(2)
    mini_data_df['time_gap_booking_prev2'] = data_df['booking_date_in_seconds'] - data_df[
        'prev2_booking_date_in_seconds']

    data_df['next2_booking_date_in_seconds'] = data_df.groupby('memberid')['booking_date_in_seconds'].shift(-2)
    mini_data_df['time_gap_booking_next2'] = data_df['next2_booking_date_in_seconds'] - data_df[
        'booking_date_in_seconds']

    # Compute time gap between current checkin date and second last checkin date/next-to-next checkin date (shift 2)
    # for each memberid and add it to mini_data
    data_df['prev2_checkin_date_in_seconds'] = data_df.groupby('memberid')['checkin_date_in_seconds'].shift(2)
    mini_data_df['time_gap_checkin_prev2'] = data_df['checkin_date_in_seconds'] - data_df[
        'prev_checkin_date_in_seconds']

    data_df['next2_checkin_date_in_seconds'] = data_df.groupby('memberid')['checkin_date_in_seconds'].shift(-2)
    mini_data_df['time_gap_checkin_next2'] = data_df['next2_checkin_date_in_seconds'] - data_df[
        'checkin_date_in_seconds']

    return mini_data_df


def inter_visit_features(data_df, mini_data_df):
    """
    For each member, compute:
    1. Difference in days_stay between current and previous/next visit
    2. Difference in roomnights between current and previous/next visit
    3. Difference in days_advance_booking between current and previous/next visit
    :param data_df: full_data
    :param mini_data_df: mini_data
    :return: modified mini_data
    """
    # Compute more information on previous and next visits (for numerical columns)
    for col in ['days_stay', 'roomnights', 'days_advance_booking']:
        data_df['prev_' + col] = data_df.groupby('memberid')[col].shift(1)
        mini_data_df['prev_diff_' + col] = data_df[col] - data_df['prev_' + col]

        data_df['next_' + col] = data_df.groupby('memberid')[col].shift(-1)
        mini_data_df['next_diff_' + col] = data_df['next_' + col] - data_df[col]

    # Compute more information on previous and next visits (for categorical columns)
    for col in ['channel_code', 'room_type_booked_code', 'resort_type_code', 'main_product_code']:
        data_df['prev_' + col] = data_df.groupby('memberid')[col].shift(1)
        mini_data_df['prev_diff_' + col] = (data_df[col] == data_df['prev_' + col]).astype(int)

        data_df['next_' + col] = data_df.groupby('memberid')[col].shift(-1)
        mini_data_df['next_diff_' + col] = (data_df[col] == data_df['next_' + col]).astype(int)

    return mini_data_df


def pivot_features(data_df, mini_data_df):
    """
    Pivot the data on row as memberid and column as each of ['resort_id', 'checkin_date_year', 'resort_type_code',
    'room_type_booked_code'] in isolation
    :param data_df: full_data
    :param mini_data_df: mini_data
    :return: modified mini_data
    """
    # Create pivots on memberid and a host of other columns and add it to mini_data
    for col in ['resort_id', 'checkin_date_year', 'resort_type_code', 'room_type_booked_code']:
        gdf = pd.pivot_table(data_df, index='memberid', columns=col, values='reservation_id', aggfunc='count',
                             fill_value=0).reset_index()
        mini_data_df = pd.merge(mini_data_df, gdf, on='memberid', how='left')

    return mini_data_df


def ratio_features(data_df):
    """
    Take ratio of values of two columns and present it as a new feature
    :param data_df: full_data
    :return: modified full_data
    """
    for col1, col2 in [["roomnights", "days_stay"]]:
        data_df[col1 + "_ratio_" + col2] = data_df[col1] / data_df[col2]

    return data_df


def concatenated_features(data_df):
    """
    Concatenate features and find the total count of reservations grouped by these features
    :param: full_data
    :return: modified full_data
    """
    for col in [
        "memberid", ["resort_id", 'checkin_date'],
        ["resort_id", "checkout_date"],
        ["state_code_residence", "checkin_date"],
        ["memberid", "checkin_date_year"],
        ["memberid", "checkin_date_month"],
        ["resort_id", "checkin_date_year"],
        ["resort_id", "checkin_date_month"],
        ["resort_id", "checkin_date_year", "checkin_date_month"],

        ["resort_id", "state_code_residence", "checkin_date"],
        ["resort_id", "state_code_residence", "checkout_date"],

        ["resort_id", "checkin_date_year", "checkin_date_week"],
        ["resort_id", "state_code_residence", "checkin_date_year", "checkin_date_week"],
        ["resort_id", "state_code_residence", "checkin_date_year", "checkin_date_month"],

        ["booking_date", "checkin_date"],
        ["booking_date", "checkin_date", "resort_id"]]:
        if not isinstance(col, list):
            col = [col]
        col_name = "_".join(col)
        gdf = data_df.groupby(col)["reservation_id"].count().reset_index()
        gdf.columns = col + [col_name + "_count"]
        data_df = pd.merge(data_df, gdf, on=col, how="left")

    return data_df
