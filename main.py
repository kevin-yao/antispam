import os
import pandas as pd
import numpy as np
import matplotlib as plt
import xgboost as xgb
from pkg.lib.model import Model
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import mode


def normalize(df, normalizedFeature):
    le = LabelEncoder()
    for feature in normalizedFeature:
        df[feature] = le.fit_transform(df[feature])

def isBanned(df):
    if df.status == 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    train_file = os.getcwd() + '/data/users_20170801.txt' 
    df = pd.read_table(train_file, header = 0 )
    normalizedFeature = ['os_name', 'gender', 'status', 'looking_for_gender']
    normalize(df, normalizedFeature)
    df['label'] = df.apply(isBanned, axis=1)
    df['mobile_prefix'] = pd.to_numeric(df['mobile_prefix'], errors = 'coerce')
    df.fillna(value=0, inplace=True)
    df['mobile_prefix'] = df['mobile_prefix'].astype(int)
    row_num = df.shape[0]
    '''
    samples = df.drop(['user_id', 'status', 'label'], axis=1)
    labels = df['label']
    train_samples_num = row_num*3/4
    train_samples = samples[0: train_samples_num]
    train_labels = labels[0: train_samples_num]
    test_samples = samples[train_samples_num : ]
    test_labels = labels[train_samples_num : ]
    '''
    train_data = df[0 : row_num*3/4]
    test_data = df[row_num*3/4 : ]
    label = 'label'
    model = Model()
    print "#################################################################################################"
    print "Logistic Regression"
    lr = LogisticRegression()
    feature = ['os_name', 'shared_device', 'contact_list', 'gender', 'looking_for_gender', 'age', 'search_min_age', 
    'search_max_age', 'mobile_prefix', 'given_likes', 'given_dislikes', 'received_likes', 'received_dislikes']
    model.classification_model(lr,train_data,feature,label)
    model.test(lr, test_data, feature, label)

    print "#################################################################################################"
    print "Decision Tree"
    dt = DecisionTreeClassifier()
    model.classification_model(dt, train_data,feature,label)
    model.test(dt, test_data, feature, label)

    print "#################################################################################################"
    print "Random Forest"
    rf = RandomForestClassifier(n_estimators=100)
    model.classification_model(rf, train_data, feature, label)
    model.test(rf, test_data, feature, label)

    print "#################################################################################################"
    print "decrease the impact of overfitting"
    rf2 = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
    model.classification_model(rf2, train_data, feature, label)
    model.test(rf2, test_data, feature, label)

    print "#################################################################################################"
    print "GBM MODEL"
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
    model.classification_model(gbm, train_data, feature, label)
    model.test(gbm, test_data, feature, label)

    print "#################################################################################################"
    print "XGBoost MODEL"
    xgb = XGBClassifier(
     learning_rate =0.2,
     n_estimators=100,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)
    model.classification_model(xgb, train_data, feature, label)
    model.test(xgb, test_data, feature, label)


