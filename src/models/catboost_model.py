import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
import zipfile 

def get_submission_csv():
    model = CatBoostClassifier()
    model.load_model("./models/model_catboost") # change model name here

    data = pd.read_csv(zipfile.ZipFile("data/processed/train.zip").open("train.csv"))
    category = pd.factorize(data["Category"], sort = True)

    X_test = pd.read_csv(zipfile.ZipFile("data/processed/test.zip").open("test.csv"))
    crime_id = X_test["Id"]
    X_test = X_test.drop("Id", axis = 1)
    cat_features = np.where(X_test.dtypes != float)[0] # get features that aren't floats
    test_dataset = Pool(data = X_test, cat_features = cat_features)
    pred_proba = model.predict_proba(test_dataset)

    # Save results
    result = pd.DataFrame(pred_proba)
    result.insert(0, "Id", crime_id)
    column_names = np.insert(category[1], 0, "Id")
    result.columns = column_names
    result.to_csv("./submissions/submission_catboost_2.csv", index = False)


if __name__== "__main__":
    # convert zip csv to df    
    data = pd.read_csv(zipfile.ZipFile("data/processed/train.zip").open("train.csv"))
    y = data.Category # get the class labels
    X = data.drop("Category", axis=1) # all features without class labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # split test and train df
    cat_features = np.where(X.dtypes != float)[0] # get features that aren't floats

    train_dataset = Pool(data = X_train, label = y_train, cat_features = cat_features)

    eval_dataset = Pool(data = X_test, label = y_test, cat_features = cat_features)

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(l2_leaf_reg = 5, learning_rate = 0.01, logging_level = "Verbose", iterations = 10000, depth = 6, random_seed = 42,
                                loss_function = "MultiClass")

    model.fit(train_dataset) # Fit model
    preds_class = model.predict(eval_dataset) # Get predicted classes
    # Get predicted probabilities for each class
    preds_proba = model.predict_proba(eval_dataset)
    # Get predicted RawFormulaVal
    preds_raw = model.predict(eval_dataset, prediction_type = "RawFormulaVal")
    
    print(model.get_best_score())
    # model.save_model("./models/model_catboost") # save the model
    # get_submission_csv() # get submission csv to Kaggle