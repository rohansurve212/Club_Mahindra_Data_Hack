Club Mahindra Data Hack
==============================

## Problem Statement
Given around ~300,000 reservations across 32 different hotels/holiday homes of Club Mahindra (CM) the objective is to predict the average spend of customers on food and beverages (FnB) per room per night stay at the hotel against the booking. A wide variety of attributes on the reservation were provided which includes `booking_date`, `checkin_date`, `checkout_date` etc. Please visit the competition homepage for more information on the problem statement and the dataset. (Source: https://datahack.analyticsvidhya.com/contest/club-mahindra-dataolympics/)

## Approach
### Feature Engineering
My approach is pretty straightforward which mainly revolves around feature engineering and feature selection. I tried many different combination of features and found the below three feature sets to be most useful for this contest.

##### 1. Features on memberid

* There were more than 100,000 unique member ids i.e. unique customers present in the whole dataset. I have created a variety of different aggregated features on memberid, which would later prove to be the most important features. This also makes intuitive sense, as with customer level features the model could get additional information about the customers past and present behavior and more importantly relate similar customers in one way or the other.

* I created a pool of such features (grouped by memberid) which include:

    * total number of reservations
    * total number of reservation at a particular resort
    * total duration of stay against each reservation
    * total number of unique resorts stayed at
    * average duration for each trip taken
    * average duration of booking a trip in advance
    * average number of people the member has travelled with in the past (total_pax)
    * cumulative count of member bookings
    * cumulative sum of days stayed at CM resorts  
    * total number of reservations in different type of rooms (room_type_booked_code)
    * total number of reservations in different holiday sessions (season_holidayed_code)
    * total number of reservations in different states (state_code_resort)
    * total number of reservations in different type of product categories (main_product_code)
    * etc.

##### 2. Temporal Features

* These are the second most imporant feature in my pool of features. Temporal features almost always helps boosted trees as most of the time these models can leverage the cross-sectional correlation of the data (e.g. interaction between different features at a given observation level) but there is no way for the model to tackle the time dimension (e.g. latent interaction between two or more observations recorded at different time points). By infusing these featues explicitly - the model can also learn the cross-time correlation e.g. how booking of member in the past affects the nature of booking of a member at present. This is very important.

* The temporal features that I considered are:

    For each member, compute:
    * time gap between current booking date and previous booking date/next booking date
    * time gap between current checkin date and previous checkin date/next checkin date
    * time gap between current checkout date and previous checkout date/next checkout date

    For each member, compute:
    * difference in duration of stay between current and previous/next visit
    * difference in duration of advanced booking between current and previous/next visit

    For each member-resort combination, compute:
    * time gap between current booking date and previous booking date/next booking date
    * time gap between current checkin date and previous checkin date/next checkin date
    
##### 3. GroupBy Multiple Features

* Another major set of features were composed by taking multiple features and finding the total count of reservations grouped by these features

##### 4. Ratio features

* I created a few ratio features. Some of those are:
    * Ratio of duration of stay to number of roomnights
    * Ratio of number of children to number of adults
    * Ratio of duration of stay to duration of advance booking

### Modeling
* I have 3 boosted trees ensemble models in total:

    * AdaBoost Regressor
    * XGBoost Regressor 
    * Extra Trees Regressor

* I also stacked the above three models and blended them using a Random Forest Regressor.

* I compared the scores of all the four models (three ensembles and the blender) and used the model with best score to predict on test set.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------
