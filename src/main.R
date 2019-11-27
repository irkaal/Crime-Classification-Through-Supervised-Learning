library(data.table)
library(reticulate)
source_python('src/utilities/data_cleaning.py')
source_python('src/utilities/outlier_handler.py')


# Load train data
path <- unzip('data/raw/sf-crime.zip', 'train.csv')
train_data <- fread(path); invisible(file.remove(path))


# Pre-process data
result_tuple <- handle_outliers(train_data) 
train_data <- result_tuple[[1]]
avg_XY <- result_tuple[[2]]

train_data <- main_clean(train_data, center_scale = F)

train_data$Descript <- train_data$Resolution <- NULL
train_data <- as.data.table(train_data)


# Prepare data for training
y_train <- as.numeric(factor(train_data$Category)) - 1
X_train <- train_data[, -which(names(train_data) == 'Category'), with = F]


# TODO: Train best model


# Prediction

# Load and clean test data
path <- unzip('data/raw/sf-crime.zip', 'test.csv')
test_data <- fread(path); invisible(file.remove(path))
test_data <- handle_outliers(test_data, avg_XY)[[1]] 
test_data <- main_clean(test_data, center_scale = F)
test_data <- as.data.table(test_data)
