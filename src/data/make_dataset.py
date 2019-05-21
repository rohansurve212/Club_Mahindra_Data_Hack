# Library imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def slice_data(raw_data):
    """ Pre-processes raw data (../data/raw) to turn it into
        interim data ready to be processed (saved in ../data/interim).
    """
    # Set random number seed
    np.random.seed(123)

    # Convert the dataframe into an array
    df_array = np.array(raw_data)
    columns = raw_data.columns.tolist()

    # shuffle the instances within the array
    shuffled_array = np.random.permutation(df_array)

    # take the first 200k instances of the shuffled array
    sliced_array = shuffled_array[:60000]

    # convert it into a pandas dataframe to further pre-process it
    interim_data = pd.DataFrame(data=sliced_array,
                                columns=columns)

    interim_data.to_csv(
        r'C:\Users\visha\Documents\GitHub\Club_Mahindra_Data_Hack\Club_Mahindra_Data_Hack\data\interim\interim_data.csv',
        index=None, header=True)

    print(interim_data.shape)

    return interim_data


def split_train_val_test(interim_data, test_size=0.2, val_size=0.2):
    """
    Split the data in a specified proportion
    :param dataframe(positional): original dataset
    :param test_size(keyword parameter): ratio of size of test set to original set
    :param val_size (keyword parameter): ratio of size of validation set to non-test set(dataset that remains after
    taking out test set from original set):
    :return: tuple of modified train, validation and test datasets
    """
    original_train, test = train_test_split(interim_data, test_size=test_size, random_state=123)
    train, val = train_test_split(original_train, test_size=val_size, random_state=123)

    train.to_csv(
        r'C:\Users\visha\Documents\GitHub\Club_Mahindra_Data_Hack\Club_Mahindra_Data_Hack\data\interim\train.csv',
        index=None, header=True)
    val.to_csv(
        r'C:\Users\visha\Documents\GitHub\Club_Mahindra_Data_Hack\Club_Mahindra_Data_Hack\data\interim\val.csv',
        index=None, header=True)
    test.to_csv(
        r'C:\Users\visha\Documents\GitHub\Club_Mahindra_Data_Hack\Club_Mahindra_Data_Hack\data\interim\test.csv',
        index=None, header=True)

    print('Training set has shape',train.shape)
    print('Validation set has shape', val.shape)
    print('Test set has shape', test.shape)

    return train, val, test

