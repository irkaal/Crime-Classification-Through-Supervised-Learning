setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source('misc.R')
source('pre_process.R')
loadPackages(c('data.table', 'caret', 'xgboost', 'plyr', 'xlsx', 'RLightGBM', 'Matrix', 'SparseM', 'LiblineaR'), quietly = T)


# 1. Data Pre-processing and feature engineering

# Load train data
path <- unzip('data/sf-crime.zip', 'train.csv')
train <- fread(path); invisible(file.remove(path))

# Clean data
train <- preProcess(train, centerScale = F) # Set centerScale to true for SVM

# Prepare data for training
y_train <- factor(make.names(as.numeric(factor(train$Category))))
x_train <- train[, -1]

# Correlation Plot
# corr_mat <- cor(as.matrix(x_train))
# corrplot::corrplot(corr_mat)

# Check for highly correlated features
# highCorr <- findCorrelation(corr_mat, cutoff = 0.5)
# colnames(x_train)[highCorr]

# Downsampling test
# index <- y_train %nin% c('X15', 'X23', 'X30', 'X34')
# y_train <- droplevels(y_train[index])
# x_train <- x_train[index,]
# ds <- downSample(x = x_train, y = y_train)
# x_train <- ds[, -68]
# y_train <- factor(ds$Class)



# Parameter tuning and feature selection

#
# xgbtree_3_2_3_2
#
system.time({
  set.seed(2019)
  xgbtree_3_2_3_2 <- train(x = x_train, 
                   y = y_train,
                   method = 'xgbTree', 
                   metric = "logLoss", 
                   trControl = trainControl(
                     method = 'adaptive_cv', 
                     number = 3, 
                     repeats = 2, 
                     adaptive = list(min = 3, alpha = 0.05, method = 'gls', complete = T),
                     classProbs = T, summaryFunction = mnLogLoss,
                     verboseIter = T, returnData = F, returnResamp = 'none', allowParallel = F, search = 'random'), 
                   tuneLength = 2)
})
write.xlsx(data.frame(xgbtree_3_2_3_2$bestTune), file = "data/param_tuning/xgbtree_3_2_3_2.xlsx", sheetName = "sheet1", row.names = F)
if (nrow(xgbtree_3_2_3_2$results)) write.xlsx(data.frame(xgbtree_3_2_3_2$results), file = "data/param_tuning/xgbtree_3_2_3_2.xlsx", sheetName = "sheet2", append = T, row.names = F)


#
# lgbm_3_2_3_2
#
system.time({
  set.seed(2019)
  lgbm_3_2_3_2 <- train(x = x_train, 
                        y = y_train, 
                        method = caretModel.LGBM(), 
                        trControl = trainControl(
                          method = 'adaptive_cv', 
                          number = 3, 
                          repeats = 2, 
                          adaptive = list(min = 3, alpha = 0.05, method = 'gls', complete = T),
                          classProbs = T, summaryFunction = mnLogLoss,
                          verboseIter = T, returnData = F, returnResamp = 'none', allowParallel = F, search = 'random'), 
                        metric = 'logLoss',
                        verbosity = -1,
                        tuneLength = 2)
})
write.xlsx(data.frame(lgbm_3_2_3_2$bestTune), file = "data/param_tuning/lgbm_3_2_3_2.xlsx", sheetName = "sheet1", row.names = F)
if (nrow(lgbm_3_2_3_2$results)) write.xlsx(data.frame(lgbm_3_2_3_2$results), file = "data/param_tuning/lgbm_3_2_3_2.xlsx", sheetName = "sheet2", append = T, row.names = F)

#
# lgbm_5_5_5_2
#
system.time({
  set.seed(5552)
  lgbm_5_5_5_2 <- train(x = data.frame(idx = 1:nrow(x_train)), # index 
                       y = y_train, 
                       matrix = Matrix(as.matrix(x_train), sparse = T),
                       method = caretModel.LGBM.sparse(), 
                       trControl = trainControl(
                         method = 'adaptive_cv', 
                         number = 5, 
                         repeats = 5, 
                         adaptive = list(min = 5, alpha = 0.05, method = 'gls', complete = T),
                         classProbs = T, summaryFunction = mnLogLoss,
                         verboseIter = T, returnData = F, returnResamp = 'none', allowParallel = F, search = 'random'),
                       metric = 'logLoss', 
                       verbosity = -1,
                       tuneLength = 2)
})
write.xlsx(data.frame(lgbm_5_5_5_2$bestTune), file = "data/param_tuning/lgbm_5_5_5_2.xlsx", sheetName = "sheet1", row.names = F)
if (nrow(lgbm_5_5_5_2$results)) write.xlsx(data.frame(lgbm_5_5_5_2$results), file = "data/param_tuning/lgbm_5_5_5_2.xlsx", sheetName = "sheet2", append = T, row.names = F)

#
# lgbm_5_5_5_5
#
system.time({
  set.seed(5555)
  lgbm_5_5_5_5 <- train(x = data.frame(idx = 1:nrow(x_train)), # index 
                        y = y_train, 
                        matrix = Matrix(as.matrix(x_train), sparse = T),
                        method = caretModel.LGBM.sparse(), 
                        trControl = trainControl(
                          method = 'adaptive_cv', 
                          number = 5, 
                          repeats = 5, 
                          adaptive = list(min = 5, alpha = 0.05, method = 'gls', complete = T),
                          classProbs = T, summaryFunction = mnLogLoss,
                          verboseIter = T, returnData = F, returnResamp = 'none', allowParallel = F, search = 'random'),
                        metric = 'logLoss', 
                        verbosity = -1,
                        tuneLength = 5)
})
write.xlsx(data.frame(lgbm_5_5_5_5$bestTune), file = "data/param_tuning/lgbm_5_5_5_5.xlsx", sheetName = "sheet1", row.names = F)
if (nrow(lgbm_5_5_5_5$results)) write.xlsx(data.frame(lgbm_5_5_5_5$results), file = "data/param_tuning/lgbm_5_5_5_5.xlsx", sheetName = "sheet2", append = T, row.names = F)


# L2 SVM with coordinate gradient descent
system.time({
  set.seed(2019)
  lsvm <- LiblineaR(data = x_train, target = y_train, verbose = F, 
                               cross = 5, findC = T)
})
write.xlsx(data.frame(lsvm), file = "data/param_tuning/lsvm_5.xlsx", sheetName = "sheet1", row.names = F)

# CV Test
tryTypes = c(0:7)
tryCosts = c(1000, 1, 0.001)
bestCost = NA
bestAcc = 0
bestType = NA
system.time({
  for (ty in tryTypes) {
    for (co in tryCosts) {
      set.seed(2019)
      acc = LiblineaR(data = x_train, target = y_train, type = ty, cost = co, bias = 1, cross = 5, verbose = FALSE)
      cat('Results for C =', co, ' : ', acc, ' accuracy.\n', sep = '')
      if (acc > bestAcc) {
        bestCost = co
        bestAcc = acc
        bestType = ty
      }
    }
  }
})
cat('Best model type is:', bestType, '\n')
cat('Best cost is:', bestCost, '\n')
cat('Best accuracy is:', bestAcc, '\n')
write.xlsx(data.frame(BestType = bestType, BestCost = bestCost, BestAcc = bestAcc), file = "data/param_tuning/lsvm_all_1113.xlsx", sheetName = "sheet1", row.names = F)


