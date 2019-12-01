import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import pickle
import zipfile

if __name__== "__main__":
    data = pd.read_csv(zipfile.ZipFile("data/processed/train.zip").open("train.csv"))
    category = pd.factorize(data["Category"], sort = True)
    y = data.Category
    X = data.drop("Category", axis=1)
    labels = list(set(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # training a DescisionTreeClassifier 
    # dtree_model = DecisionTreeClassifier(max_depth = 30).fit(X_train, y_train) 
    # dtree_predictions = dtree_model.predict(X_test) 
    # print(classification_report(y_test, dtree_predictions, labels=labels))

    # Cross validation:
    parameters = {"max_depth": range(15, 30)}
    clf = GridSearchCV(tree.DecisionTreeClassifier(random_state = 42), parameters, n_jobs = 4, cv = 5, verbose = 5)
    clf.fit(X=X, y=y)
    tree_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)
    dtree_predictions = tree_model.predict(X_test)
    print(classification_report(y_test, dtree_predictions, labels = labels))
    print("Coefficient of Determination: " + str(tree_model.score(X_test, y_test)))
    
    # Save model
    # pkl_filename = "./models/dt_model.pkl"
    # with open(pkl_filename, "wb") as file:
    #     pickle.dump(tree_model, file)

    # Save results
    X_test = pd.read_csv(zipfile.ZipFile("data/processed/test.zip").open("test.csv"))
    crime_id = X_test["Id"]
    X_test = X_test.drop("Id", axis = 1)
    pred_proba = tree_model.predict_proba(X_test)
    result = pd.DataFrame(pred_proba)
    result.insert(0, "Id", crime_id)
    column_names = np.insert(category[1], 0, "Id")
    result.columns = column_names
    result.to_csv("./submissions/submission_dt.csv", index = False)