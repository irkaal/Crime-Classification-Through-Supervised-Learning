
# Data cleaning wrapper
preProcess <- function(dataset, centerScale = F) {
  source('misc.R')
  source('outlier_handler.R')
  loadPackages(c('data.table', 'reticulate'))
  # Handle Outliers
  dataset <- handleOutliers(dataset, write = F)
  # Main cleaning task
  source_python('dataCleaning.py')
  dataset <- as.data.table(mainClean(dataset, centerScale))
  return(dataset)
}

# For use with Python rpy2
# Return NULL as rpy2 is unable to convert R dataframe back to Pandas dataframe
preProcessPy <- function(dataset) {
  source('misc.R')
  source('outlier_handler.R')
  loadPackages('data.table')
  dataset <- as.data.table(dataset)
  # Handle Outliers
  dataset <- handleOutliers(dataset, write = T)
  return(NULL)
}
