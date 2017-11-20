import pandas as pd
import numpy as np
import matplotlib as plt
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import mode


class Model(object):
    #Generic function for making a classification model and accessing performance:
    def classification_model(self, model, train_data, features, label):
        #Fit the model:
        model.fit(train_data[features], train_data[label])

        #Make predictions on training set:
        predictions = model.predict(train_data[features])
        predprob = model.predict_proba(train_data[features])[:,1]

        print(metrics.classification_report(train_data[label], predictions))
        print(metrics.confusion_matrix(train_data[label], predictions))
        #Print accuracy
        #accuracy = metrics.accuracy_score(data[outcome], predictions)
        #recall = metrics.recall_score(data[outcome], predictions)
        #print "Accuracy : %s" % "{0:.3%}".format(accuracy)
        #print "Recall : %s" % "{0:.3%}".format(recall)
        print "AUC Score (Train): %f" % metrics.roc_auc_score(train_data[label], predprob)

        #Perform k-fold cross-validation with 5 folds
        kf = KFold(train_data.shape[0], n_folds=5)
        error = []
        for train, test in kf:
            # Filter training data
            train_predictors = (train_data[features].iloc[train,:])

            # The target we're using to train the algorithm.
            train_target = train_data[label].iloc[train]

            # Training the algorithm using the predictors and target.
            model.fit(train_predictors, train_target)
            #Record error from each cross-validation run
            error.append(model.score(train_data[features].iloc[test,:], train_data[label].iloc[test]))

        print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

        #Fit the model again so that it can be refered outside the function:
        model.fit(train_data[features],train_data[label]) 

    def test(self, model, test_data, features, label):
        #Make predictions on training set:
        predictions = model.predict(test_data[features])
        #predprob = model.predict_proba(test_data[label])[:,1]
        print("Testing result:")
        print(metrics.classification_report(test_data[label], predictions))
        print(metrics.confusion_matrix(test_data[label], predictions))