
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


def train_xgb(df_preprocessed, df_target):
    """
    Train an XGBoost Regressor on the data
    :param df_preprocessed: features
    :param df_target: target
    :return: a tuple of best estimator and best estimator score
    """
    xgb_reg = XGBRegressor(nthread=4,
                           objective='reg:linear',
                           learning_rate=0.02,   # so called `eta` value
                           max_depth=10,
                           min_child_weight=1,
                           gamma=3,
                           subsample=1.0,
                           colsample_bytree=0.35)

    param_grid = {'n_estimators': [1000]}

    gridsearch_xgb = GridSearchCV(xgb_reg, param_grid, cv=3, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    gridsearch_xgb.fit(df_preprocessed, df_target)

    # save the model to disk
    # xgb_filename = r'models\xgboost_model.sav'
    # pickle.dump(gridsearch_xgb, open(xgb_filename, 'wb'))
    print(np.sqrt(-gridsearch_xgb.best_score_))

    return gridsearch_xgb.best_estimator_, np.sqrt(-gridsearch_xgb.best_score_)


def train_adb(df_preprocessed, df_target):
    """
    Train an AdaBoost Regressor on the data
    :param df_preprocessed: features
    :param df_target: target
    :return: a tuple of best estimator and best estimator score
    """
    adb_reg = AdaBoostRegressor(random_state=123)

    param_grid = {'n_estimators': [1000],
                  'learning_rate': [0.02]}

    gridsearch_adb = GridSearchCV(adb_reg, param_grid, cv=3, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    gridsearch_adb.fit(df_preprocessed, df_target)

    # save the model to disk
    # xgb_filename = r'models\xgboost_model.sav'
    # pickle.dump(gridsearch_xgb, open(xgb_filename, 'wb'))
    print(np.sqrt(-gridsearch_adb.best_score_))

    return gridsearch_adb.best_estimator_, np.sqrt(-gridsearch_adb.best_score_)


# def train_hgb(df_preprocessed, df_target):
#     """
#     Train a Hist Gradient Boosting Regressor on the data
#     :param df_preprocessed: features
#     :param df_target: target
#     :return: a tuple of best estimator and best estimator score
#     """
#     hgb_reg = HistGradientBoostingRegressor(random_state=123)
#
#     param_grid = {'n_estimators': [1000],
#                   'learning_rate': np.arange(0.01, 0.1, 0.01),
#                   'l2_regularization': np.arange(0, 1, 0.2),
#                   'max_depth': np.arange(5, 10, 1),
#                   'max_iter': np.arange(100, 500, 100)}
#
#     gridsearch_hgb = GridSearchCV(hgb_reg, param_grid, cv=4, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
#     gridsearch_hgb.fit(df_preprocessed, df_target)
#
#     # save the model to disk
#     # xgb_filename = r'models\xgboost_model.sav'
#     # pickle.dump(gridsearch_xgb, open(xgb_filename, 'wb'))
#     print(np.sqrt(-gridsearch_hgb.best_score_))
#
#     return gridsearch_hgb.best_estimator_, np.sqrt(-gridsearch_hgb.best_score_)


def train_etr(df_preprocessed, df_target):
    """
    Train an Extra Trees Regressor on the data
    :param df_preprocessed: features
    :param df_target: target
    :return: a tuple of best estimator and best estimator score
    """
    etr_reg = ExtraTreesRegressor(random_state=123)

    param_grid = {'n_estimators': [1000],
                  'max_depth': [10],
                  'min_samples_split': [0.1],
                  'min_samples_leaf': [0.01]}

    gridsearch_etr = GridSearchCV(etr_reg, param_grid, cv=3, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    gridsearch_etr.fit(df_preprocessed, df_target)

    # save the model to disk
    # xgb_filename = r'models\xgboost_model.sav'
    # pickle.dump(gridsearch_xgb, open(xgb_filename, 'wb'))
    print(np.sqrt(-gridsearch_etr.best_score_))

    return gridsearch_etr.best_estimator_, np.sqrt(-gridsearch_etr.best_score_)


def stack_models(df_preprocessed, df_target, model_1, model_2, model_3):
    """
    Stack all the three models and blend their predictions in a Random Forest Regressor; Train and evaluate the
    Blender's performance on Validation set
    :param df_preprocessed: Validation Set Features,
    :param df_target: Validation Set Target
    :param model_1: 1st model to be added to the stack
    :param model_2: 2nd model to be added to the stack
    :param model_3: 3rd model to be added to the stack
    :return: A tuple of Trained stacked Model and the Mean of it's Cross-validated RMSE
    """

    # Bring together the best estimators of all the three ML models and the deep neural network model
    estimators = [model_1, model_2, model_3]

    # Creating training set for the Stacker/Blender
    stack_predictions = np.empty((df_preprocessed.shape[0], len(estimators)), dtype=np.float32)
    for index, estimator in enumerate(estimators):
        stack_predictions[:, index] = np.reshape(estimator.predict(df_preprocessed), (df_preprocessed.shape[0],))

    # Initializing the Stacker/Blender (Random Forest Regressor)
    rf_blender = RandomForestRegressor(n_estimators=20, random_state=123)

    # Evaluate the Blender on stacking set using cross-validation (# cross validation sets =3)
    val_scores = cross_val_score(rf_blender, stack_predictions, df_target, scoring='neg_mean_squared_error', n_jobs=-1)

    print(np.mean(np.sqrt(np.array(val_scores)*-1)))

    return rf_blender, np.mean(np.sqrt(np.array(val_scores)*-1))


def evaluate_on_test(best_model, test_preprocessed, test_target):
    """
    Evaluate the performance of the best model on test set
    :param best_model: The model to predict on test set
    :param test_preprocessed: Test Features
    :param test_target: Test Target
    :return: RMSE of predicted values
    """

    # Make predictions on the data
    test_prediction = best_model.predict(test_preprocessed)

    # Evaluate the predictions against the actual targets
    test_score = mean_squared_error(test_target, test_prediction)

    return np.sqrt(test_score)
