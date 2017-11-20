#!/usr/bin/python
#coding=utf-8
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt


def preprocessing(df, normalizedFeature):
    le = LabelEncoder()
    for feature in normalizedFeature:
        df[feature] = le.fit_transform(df[feature])

def isBanned(df):
    if df.status == 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    train_file = os.getcwd() + '/data/data_1.txt' 
    df = pd.read_table(train_file, header = 0 )
    #print(df.dtypes)
    #df.fillna(value=0, inplace=True)
    #print(df.loc[df['os_name'] == 'None', ['os_name']])
    #print(df.head(100))
    #print(df.loc[df['status'] == 'banned', ['user_id', 'status']].head(50))
    normalizedFeature = ['os_name', 'gender', 'status', 'looking_for_gender']
    preprocessing(df, normalizedFeature)
    df['label'] = df.apply(isBanned, axis=1)

    df['mobile_prefix'] = pd.to_numeric(df['mobile_prefix'], errors = 'coerce')
    df.fillna(value=0, inplace=True)
    df['mobile_prefix'] = df['mobile_prefix'].astype(int)
    #print(df.dtypes)
    #df.fillna(0, inplace=True)
    #print(df.loc[df['os_name'] == 0, ['os_name']])

    #print(df.loc[df['status'] == 0, ['user_id', 'status']].head(50))
    samples = df.drop(['user_id', 'status', 'label'], axis=1).as_matrix()
    #print(samples[0:100])
    #labels = df[['label']].copy()
    labels = df['label'].as_matrix()
    #print(labels[0:100])
    train_samples = samples[0: 50000]
    train_labels = labels[0: 50000]
    test_samples = samples[50000 : ]
    test_labels = labels[50000 : ]    
    #print(test_labels.loc[test_labels['label'] == 1].count())
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    rf.fit(train_samples, train_labels)
    predicted = rf.predict(test_samples)
    expected = test_labels
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    
    feature_names = list()
    with open('doc/feature.txt') as fp:
        for line in fp:
            feature_names.append(line)

    feature_importance = rf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, [feature_names[i] for i in sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
