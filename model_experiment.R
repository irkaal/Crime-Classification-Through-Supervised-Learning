setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source('misc.R')
source('pre_process.R')
loadPackages(c('data.table', 'slam', 'caret', 'xgboost', 'plyr', 'h2o', 'xlsx', 'parallel', 'doParallel'), quietly = T)
registerDoParallel(2)



# 1. Data Pre-processing and feature engineering

# Load train data
# TODO: REMOVE
train <- fread('data/clean.csv')
# path <- unzip('data/sf-crime.zip', 'train.csv')
# train <- fread(path); invisible(file.remove(path))

# Clean data
# train <- preProcess(train)

# Prepare data for training
y_train <- factor(make.names(as.numeric(factor(train$Category))))
x_train <- train[, -68]
rm(train)

# Correlation Plot
# corr_mat <- cor(x_train)
# corrplot::corrplot(corr_mat)

# Remove highly correlated features
# highCorr <- findCorrelation(corr_mat, cutoff = 0.75)
# predictors(highCorr)
# x_train <- x_train[, -highCorr]

# Downsampling test
# ds <- downSample(x = x_train, y = y_train)
# x_train <- ds[, -68]
# y_train <- factor(ds$Class)
# rm(ds)



# Parameter tuning and feature selection

# Random Control - Random Hyperparameter Search
randCtrl <- trainControl(method = 'repeatedcv', 
                         number = 10,
                         repeats = 10,
                         classProbs = T,
                         summaryFunction = mnLogLoss,
                         search = 'random') 

# Adaptive Resampling - Racing-type algorithm
# Generalized Least Squares Model
adaptGlsCtrl <- trainControl(method = 'adaptive_cv',
                             number = 10, 
                             repeats = 10,
                             adaptive = list(min = 5, alpha = 0.05, 
                                             method = 'gls', complete = T),
                             classProbs = T,
                             summaryFunction = mnLogLoss,
                             search = 'random')
# Bradley-Terry model
adaptBTCtrl <- trainControl(method = 'adaptive_cv',
                            number = 10, 
                            repeats = 10,
                            adaptive = list(min = 5, alpha = 0.05, 
                                            method = 'BT', complete = T),
                            classProbs = T,
                            summaryFunction = mnLogLoss,
                            search = 'random')



# XGBoost - eXtreme Gradient Boosting

# Tree
set.seed(2019)
xgbTreeModel <- train(x = x_train, 
                      y = y_train,
                      method = 'xgbTree', 
                      metric = "logLoss", 
                      trControl = adaptBTCtrl, 
                      tuneLength = 15)
write.xlsx(data.frame(xgbTreeModel$bestTune), file = "data/param_tuning/xgbTreeModel.xlsx", sheetName = "sheet1", row.names = F)
if (nrow(xgbTreeModel$results)) write.xlsx(data.frame(xgbTreeModel$results), file = "data/param_tuning/xgbTreeModel.xlsx", sheetName = "sheet2", append = T, row.names = F)


# Linear Model
set.seed(2019)
xgbLinearModel <- train(x = x_train, 
                        y = y_train,
                        method = 'xgbLinear', 
                        metric = "logLoss", 
                        trControl = adaptBTCtrl, 
                        tuneLength = 15)
write.xlsx(data.frame(xgbLinearModel$bestTune), file = "data/param_tuning/xgbLinearModel.xlsx", sheetName = "sheet1", row.names = F)
if (nrow(xgbLinearModel$results)) write.xlsx(data.frame(xgbLinearModel$results), file = "data/param_tuning/xgbLinearModel.xlsx", sheetName = "sheet2", append = T, row.names = F)


# Dropouts meet Multiple Additive Regression Trees
set.seed(2019)
xgbDARTModel <- train(x = x_train, 
                      y = y_train,
                      method = 'xgbDART', 
                      metric = "logLoss", 
                      trControl = adaptBTCtrl, 
                      tuneLength = 15)
write.xlsx(data.frame(xgbDARTModel$bestTune), file = "data/param_tuning/xgbDARTModel.xlsx", sheetName = "sheet1", row.names = F)
if (nrow(xgbDARTModel$results)) write.xlsx(data.frame(xgbDARTModel$results), file = "data/param_tuning/xgbDARTModel.xlsx", sheetName = "sheet2", append = T, row.names = F)


# H2O Gradient Boosting Machines 

# Setup
# h2o.init()
# options('h2o.use.data.table' = T)
# x_train_h2o <- as.h2o(x_train)
# y_train_h2o <- factor(make.names(y_train))
# rm(list = c('x_train', 'y_train'))

# Default GBM
# set.seed(2019)
# gbmModel <- train(x = x_train_h2o, 
#                   y = y_train_h2o,
#                   method = 'gbm_h2o', 
#                   metric = "logLoss", 
#                   trControl = adaptBTCtrl, 
#                   tuneLength = 15)

# gbmPred <- predict(gbmModel, x_train_h2o)

# Save results
# fwrite(data.table(gbmPred), 'data/prediction/gbmModelPred.csv')
# write.xlsx(data.frame(gbmModel$results),
#            file = "data/param_tuning/gbmModel.xlsx", sheetName = "sheet1", row.names = F)
# write.xlsx(data.frame(gbmModel$bestTune),
#            file = "data/param_tuning/gbmModel.xlsx", sheetName = "sheet2", append = T, row.names = F)


# TODO: 
# svmLinear3
# lssvmLinear 
# lssvmPoly
# svmBoundrangeString
# svmExpoString
# svmLinear
# svmPoly
# svmRadial
# svmRadialCost
# svmRadialSigma
# svmSpectrumString
# RSofia SGD - Not caret


# TODO: 
# ranger
# Random Forest by Randomization
# Random Ferns


# TODO: 
# Sparse LDA
# Penalized LDA
# Bagged FDA


# TODO: 
# Nearest Shrunken Centroids



# 3. Prediction

# Load test data
# path <- unzip('data/sf-crime.zip', 'test.csv')
# test <- fread(path); invisible(file.remove(path))
# rm(path)

# Clean data
# test <- preProcess(test)

# Predict

# Clean up
# rm(list = ls())
