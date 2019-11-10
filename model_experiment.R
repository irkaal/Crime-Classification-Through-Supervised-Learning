
############################
# Load Libraries and Setup #
############################


source('./misc.R')
loadPackages(c(
  'data.table', 'slam', 'Matrix', 'mltools', # For performance boost
  'caret', 'xgboost', 'h2o',                 # For parameter tuning, H2O (http://h2o-release.s3.amazonaws.com/h2o/rel-yau/10/index.html)
  'foreach', 'parallel', 'doParallel',       # For parallelization
  'reticulate'                               # Python interface
))

# Setup
h2o.init()
options("h2o.use.data.table" = T)
registerDoParallel(detectCores() / 2)
source_python('./dataCleaning.py')


#################
# Data Cleaning #
#################


path <- unzip('./data/sf-crime.zip', 'train.csv')
crime_data <- fread(path); invisible(file.remove(path))
crime_data <- data.table(mainClean(crime_data))
str(crime_data)

# Split data.table into data matrix and response vector
train_data <- mltools::sparsify(crime_data[, -1])
train_label <- as.numeric(factor(crime_data$Category))


####################
# Parameter Tuning #
####################


set.seed(2019)


##########################
# XGBoost (Tree Booster) #
##########################


# Direct (No Tuning, only nrounds)
dtrain <- xgb.DMatrix(data = train_data, label = train_label)
bst <- xgboost(data = dtrain, booster = 'gbtree', objective = 'multi:softmax', num_class = 39, 
               eval_metric = 'mlogloss', nrounds = 50)
pred <- predict(bst, train_data)

# Caret
xgbControl <- trainControl(method = 'repeatedcv', number = 10, repeats = 10,
                           allowParallel = T,
                           verboseIter = T, classProbs = T,
                           savePredictions = T,  returnData = F,
                           summaryFunction = mnLogLoss)
xgbGrid <- expand.grid(max_depth = c(6),
                       nrounds = c(50, 100),
                       eta = c(0.2, 0.3, 0.4),
                       gamma = c(0, 0.25, 0.5, 0.75, 1), # Regularization
                       colsample_bytree = 1,
                       min_child_weight = 1,
                       subsample = c(0.5, 1))
xgbModel <- train(x = train_data, y = factor(make.names(train_label)),
                  method = 'xgbTree', metric = "logLoss", 
                  trControl = xgbControl, tuneGrid = xgbGrid)
print(xgbModel)
pred <- predict(xgbModel, train_data)


#############
# H20 (GBM) #
#############


h2o_train <- as.h2o(crime_data[, -1])
gbmControl <- trainControl(method = 'none')
gbmControl <- trainControl(method = 'repeatedcv', number = 10, repeats = 10,
                           allowParallel = T,
                           verboseIter = T, classProbs = T,
                           savePredictions = T,  returnData = F,
                           summaryFunction = mnLogLoss)
gbmGrid <- expand.grid(ntrees = 50, 
                       max_depth = c(5), 
                       min_rows = c(10), 
                       learn_rate = c(0.1), 
                       col_sample_rate = c(1))
gbmModel <- train(x = h2o_train, y = factor(make.names(train_label)),
                  method = 'gbm_h2o', metric = "logLoss", 
                  trControl = gbmControl, tuneGrid = gbmGrid)
print(gbmModel)
pred <- predict(gbmModel, train_data)
mean(train_data$Category == pred)

