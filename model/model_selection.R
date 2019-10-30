
compareModels <- function(train, k, m) {
  source('misc.R')
  required_packages <- c('doParallel', 'foreach', 'MASS', 'parallel', 'randomForest')
  loadPackages(required_packages)
  registerDoParallel(detectCores() - 1)
  ii <- 1:nrow(train) %% k + 1
  
  # For each 5-fold CV i, i = 1, 2, ..., m
  score <- foreach(i = 1:m, .packages = required_packages) %dopar% {
    # Randomize order
    ii <- sample(ii)
    pr_model1 <- rep(-1, length(ii))
    
    # For each fold j, j = 1, 2, ..., k    
    foreach(j = 1:k) %dopar% {
      train_data <- train[j != ii, ]
      validation_data <- train[j == ii, ]
      
      # Model 1: Random Forest with all features
      model1 <- randomForest(Survived ~ ., data = train_data)
      pr_model1[j == ii] <- predict(model1, newdata = validation_data)
    }
    
    return(list(
      model1 = mean(train$Survived == pr_model1)
    ))
  }
  
  result <- do.call(rbind, score)
  return(result)
}
