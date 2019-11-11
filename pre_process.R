
# Data cleaning wrapper
preProcess <- function(dataset) {
  source('misc.R')
  source('outlier_detection.R')
  loadPackages(c('data.table', 'reticulate'))
  # Handle Outliers
  dataset <- handleOutliers(dataset, write = F)
  # Main cleaning task
  source_python('dataCleaning.py')
  dataset <- mainClean(dataset)
  dataset <- as.data.table(dataset)
  return(dataset)
}

# For use with Python rpy2
# Return NULL as rpy2 is unable to convert R dataframe back to Pandas dataframe
preProcessPy <- function(dataset) {
  source('misc.R')
  source('outlier_detection.R')
  loadPackages('data.table')
  dataset <- as.data.table(dataset)
  # Handle Outliers
  dataset <- handleOutliers(dataset, write = T)
  return(NULL)
}
