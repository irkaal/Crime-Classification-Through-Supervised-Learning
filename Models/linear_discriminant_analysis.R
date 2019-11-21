setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Load the libraries
library(data.table)
library(caret)
library(MASS)

# Load the cleaned data
path <- unzip('../data/clean_centered_final.zip', 'clean_centered_final.csv')
train <- fread(path); invisible(file.remove(path))
y_train <- factor(make.names(train$Category))
X_train <- train[, -which(names(train) == 'Category'), with = F]
rm(train)

# Log Loss
ctrl <- trainControl(method = 'repeatedcv', number = 2, repeats = 5,
                     classProbs = T, summaryFunction = mnLogLoss,
                     verboseIter = T, allowParallel = F)
set.seed(2019)
lda_model <- train(x = X_train, y = y_train, method = 'lda', metric = 'logLoss', trControl = ctrl)

# Accuracy and Kappa
ctrl <- trainControl(method = 'repeatedcv', number = 2, repeats = 5,
                     verboseIter = T, allowParallel = F)
set.seed(2019)
lda_model_acc <- train(x = X_train, y = y_train, method = 'lda', metric = 'Accuracy', trControl = ctrl)
