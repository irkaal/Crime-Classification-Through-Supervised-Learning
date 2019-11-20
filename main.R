setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Source file and load libraries
library(data.table)
library(xgboost)
library(plyr)
library(ggplot2)
library(reticulate)
library(caret)
source('outlier_handler.R')
source_python('data_cleaning.py')

# 1. Data Pre-processing and feature engineering

# Load train data
path <- unzip('data/sf-crime.zip', 'train.csv')
train <- fread(path); invisible(file.remove(path))

# Clean data
train <- handleOutliers(train)
# To center data, set centerScale = T
train <- as.data.table(main_clean(train, center_scale = F))

train <- readRDS('data/clean_old_features.rds')

# Prepare data for training
y_train <- as.numeric(factor(train$Category)) - 1
X_train <- train[, -which(names(train) == 'Category'), with = F]

# Correlation Plot
corr_mat <- cor(as.matrix(X_train))
corrplot::corrplot(corr_mat)

# Check for highly correlated features
highCorr <- findCorrelation(corr_mat, cutoff = 0.75)
colnames(X_train)[highCorr]

# Check for near zero variance predictors
# We will ignore the results because we know that both Event and Year_2005 are binary features
nzv <- nearZeroVar(X_train, saveMetrics = F)
nzv[nzv$nzv,]

# Plot feature importance plot
dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
xgb_model <- xgboost(data = dtrain, verbose = 1, early_stopping_rounds = 100,
                     objective = 'multi:softmax', eval_metric = 'mlogloss', num_class = 39, nrounds = 1)
importance_matrix <- xgb.importance(model = xgb_model)
xgb.ggplot.importance(importance_matrix)
