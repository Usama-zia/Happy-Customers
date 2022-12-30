"""
Outliers removal
balance the dataset
Scalling the dataset
different Feature Selection Approaches
"""
import numpy as np
import scipy.stats as stats
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE
class Feat_select():
    """To clean the dataset and select features."""
    def select_full_features(dataset):
        #dataset slicing for Manual feature selection
        #selecting all features
        sliced_features = dataset.iloc[:,1:7]
        sliced_labels = dataset['Y']

        #removing outliers
        #only keep rows in dataframe with all z-scores less than absolute value of 3
        z = np.abs(stats.zscore(sliced_features))
        X_clean = sliced_features[(z<3).all(axis=1)]
        y_clean = sliced_labels[(z<3).all(axis=1)]

        #Balancing Dataset
        sm = SMOTE()
        X_clean, y_clean = sm.fit_resample(X_clean, y_clean)
        print(X_clean.shape)
        #scale and normalize the dataset
        scaling = StandardScaler()
        X_clean = scaling.fit_transform(X_clean)

        return X_clean, y_clean

    def manual_select_features(dataset):
        #dataset slicing for Manual feature selection
        #selecting feaures X1,X2,X5,X6
        sliced_features = dataset.iloc[:, [1, 2, 5, 6]]
        sliced_labels = dataset['Y']

        #removing outliers
        #only keep rows in dataframe with all z-scores less than absolute value of 3
        z = np.abs(stats.zscore(sliced_features))
        X_clean = sliced_features[(z<3).all(axis=1)]
        y_clean = sliced_labels[(z<3).all(axis=1)]

        #Balancing Dataset
        sm = SMOTE()
        X_clean, y_clean = sm.fit_resample(X_clean, y_clean)

        #scale and normalize the dataset
        scaling = StandardScaler()
        X_clean = scaling.fit_transform(X_clean)

        return X_clean, y_clean

    def stats_select_features(dataset):
        #Slicing for feature selection with stats measures
        sliced_features = dataset.iloc[:,1:7]
        sliced_labels = dataset['Y']
        sliced_features.head()

        #feature selection
        fs = SelectKBest(score_func=f_classif, k=4)
        X_selected = fs.fit_transform(sliced_features, sliced_labels)

        #Checking which features where selected
        filter = fs.get_support()
        feat = np.array(sliced_features.columns)
        print('Total Features ', feat)
        print('Selected Features for training ',feat[filter])

        #removing outliers
        #only keep rows in dataframe with all z-scores less than absolute value of 3
        z = np.abs(stats.zscore(X_selected))
        X_clean = X_selected[(z<3).all(axis=1)]
        y_clean = sliced_labels[(z<3).all(axis=1)]

        #Balancing Dataset
        sm = SMOTE()
        X_clean, y_clean = sm.fit_resample(X_clean, y_clean)

        #scale and normalize the dataset
        scaling = StandardScaler()
        X_clean = scaling.fit_transform(X_clean)

        return X_clean, y_clean