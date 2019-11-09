############################
# Load Libraries and Setup #
############################


# install.packages(c('data.table', 'Matrix', 'mltools', 'caret', 'xgboost', 'h2o', 'foreach', 'parallel', 'doParallel'))

# For performance boost
library(data.table)
library(Matrix)
library(mltools)

# For parameter tuning
library(caret)
library(xgboost)
# For setting up H2O (http://h2o-release.s3.amazonaws.com/h2o/rel-yau/10/index.html)
# H2O requires Java 64-Bit
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-yau/10/R")
library(h2o)
h2o.init()

# For parallelization
library(foreach)
library(parallel)
library(doParallel)
registerDoParallel(detectCores() - 1)

# For sourcing python files
# library(reticulate)
# source_python('./dataCleaning.py')

set.seed(2019)


#################
# Data Cleaning #
#################


# crime_data <- read.csv(unz('./data/sf-crime.zip', 'train.csv'))
# crime_data <- mainClean(crime_data)
# write.csv(crime_data, './train_clean.csv', row.names = F)


#####################
# Load Cleaned Data #
#####################


path <- unzip('./data/sf-crime-clean.zip', 'train_clean.csv')
crime_data <- fread(path); invisible(file.remove(path))

# Structure of data.table
str(crime_data)

# Split data.table into data matrix and response vector
train_data <- sparsify(crime_data[, -1])
train_label <- as.numeric(crime_data$Category)


##########################
# XGBoost (Tree Booster) #
##########################


# Direct (No Tuning, only nrounds)
dtrain <- xgb.DMatrix(data = train_data, label = train_label)
bst <- xgb.train(data = dtrain, booster = 'gbtree',
                 objective = 'multi:softmax', num_class = 39, eval_metric = 'mlogloss',
                 nrounds = 1)
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


h2otrain <- as.h2o(crime_data[, -1])
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
gbmModel <- train(x = h2otrain, y = factor(make.names(train_label)),
                  method = 'gbm_h2o', metric = "logLoss", 
                  trControl = gbmControl, tuneGrid = gbmGrid)
print(gbmModel)
pred <- predict(gbmModel, train_data)


############
# LightGBM #
############


# TODO: Setup
