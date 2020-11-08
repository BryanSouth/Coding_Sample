# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:59:38 2019

@author: Bryan
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
#from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sn
import sklearn.metrics as metrics
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report




# In[] 
def init():
    
    data = np.load('train.npy')

    df_data = pd.DataFrame(data, columns = ['slip', 'motor torque', 'contact pressure', 'class' ])
    df_x = df_data.loc[:, df_data.columns != 'class']
    df_y = df_data.loc[:, df_data.columns == 'class']
    X = df_x.values
    Y = df_y.values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    y_train = np.ravel(y_train)
    return x_train, y_train, x_test, y_test

# In[]    

def cross_val():
    
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)] 
    max_features = [1, 2, 3] 
    max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]
    min_samples_split = [2, 5, 10]
    bootstrap = [True, False]
    criterion = ['gini', 'entropy']
    min_samples_leaf = [int(x) for x in np.linspace(1, 10, num = 1)]
    
    random_grid = {'n_estimators': n_estimators,
           'max_features': max_features,
           'max_depth': max_depth,
           'min_samples_split': min_samples_split,
           'bootstrap': bootstrap,
           'criterion': criterion,
           'min_samples_leaf': min_samples_leaf}
    
    x_train, y_train, x_test, y_test = init()
    clas = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = clas, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = 5)
    rf_random.fit(x_train, y_train)
    hypers = rf_random.best_params_
    
    return hypers
    
def main():
    
    x_train, y_train, x_test, y_test = init()
    hypers = cross_val()
    clas = RandomForestClassifier(n_estimators=hypers['n_estimators'], max_depth = hypers['max_depth'], min_samples_split = hypers['min_samples_split'], 
                                  min_samples_leaf = hypers['min_samples_leaf'], max_features = hypers['max_features'], 
                                  criterion = hypers['criterion'])
    clas.fit(x_train, y_train)
    

    
    # In[]
    Y_pred = clas.predict(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))
    
    compare = np.column_stack((y_test,Y_pred))
    df = pd.DataFrame(compare, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    ax = plt.axes()
    sn.heatmap(confusion_matrix, annot=True)
    ax.set_title('Confusion Matrix of Multi-Class Classification of Mars Terrain Simulants')
    print(classification_report(y_test, Y_pred))
    
    print('f1 score is', f1_score(y_test, Y_pred, average="macro"))
    print('precision score is', precision_score(y_test, Y_pred, average="macro"))
    print('recall is', recall_score(y_test, Y_pred, average="macro")) 
    
    #Confusion Matrix of Multi-Class Classification of MMS 2mm, GRC-01,WED-720, and MMS Coarse

    
    importances = clas.feature_importances_
    indices = np.argsort(importances)
    features = ['Wheel Slip', 'Shear Stress', 'Contact Pressure']
    plt.figure()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

if __name__ == "__main__":
    main()