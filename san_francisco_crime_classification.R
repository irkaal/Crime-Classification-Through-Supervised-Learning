# Packages
library(reticulate)
library(tidyverse)
library(styler)
# style_file('San_francisco_crime_classification.R')

# Data
trainfile <- './data/train.csv'
testfile <- './data/test.csv'
if (!file.exists(trainfile) || !file.exists(testfile)) {
  print('unzipping')
  setwd('./data/')
  paths <- paste(sep = '', './data/', unzip('./sf-crime.zip', c('train.csv', 'test.csv')))
  setwd('..')
  print('done')
}

train <- read.csv(paths[1])
train$Id <- NA

test <- read.csv(paths[2])
test <- test %>%
  mutate(
    Category = NA,
    Descript = NA,
    Resolution = NA
  )
crime_data <- rbind(train, test)

# Feature Engineering:

# 1. Dates
dates_split <- unlist(str_split(crime_data$Dates, "-|[ ]|:"))
crime_data <- crime_data %>%
  mutate(
    Year = factor(dates_split[c(T, F, F, F, F, F)]),
    Month = factor(dates_split[c(F, T, F, F, F, F)]),
    Day = factor(dates_split[c(F, F, T, F, F, F)]),
    Hour = factor(dates_split[c(F, F, F, T, F, F)]),
    Minute = factor(dates_split[c(F, F, F, F, T, F)]),
    Dates = NULL
  )


# 5. PdDistrict ~ PatrolDivision
# Patrol divisions are broken down into two divisions Golden Gate Division and Metro Division which are each led by San Francisco Police Commanders.
# https://en.wikipedia.org/wiki/San_Francisco_Police_Department
metro_div <- c("CENTRAL", "INGLESIDE", "NORTHERN", "SOUTHERN", "TENDERLOIN")
golden_gate_div <- c("BAYVIEW", "MISSION", "PARK", "RICHMOND", "TARAVAL")
crime_data <- crime_data %>%
  mutate(PatrolDivision = factor(case_when(
    PdDistrict %in% metro_div ~ "Metro",
    PdDistrict %in% golden_gate_div ~ "GoldenGate"
  )))


######
# TEST
######

test_df <- crime_data[1,1:3]
source_python("./test.py")
print(myfunc(test_df))

k <- 5
n <- nrow(train)
ii <- (1:n) %% k + 1
m <- 50
accuracy_model1 <- accuracy_model2 <- accuracy_model3 <- rep(0, m)

numCores <- detectCores() - 1
registerDoParallel(numCores)

result <- foreach (i = 1:m, .packages = c('doParallel', 'MASS', 'randomForest')) %dopar% {
  
  ii <- sample(ii)
  pr_model1 <- pr_model2 <- pr_model3 <- rep(NA, n)
  
  foreach (j = 1:k) %dopar% {
  
    train_data <- train[j != ii, ]
    validation_data <- train[j == ii, ]
    
    # Model 1
    lr_full <- glm(Survived ~ ., family = binomial, data = train_data)
    lr_null <- glm(Survived ~ 1, family = binomial, data = train_data)
    lr_sub <- stepAIC(object = lr_null, scope = list(lower = lr_null, upper = lr_full), trace = F)
    prob_sub <- predict(object = lr_sub, type = 'response', newdata = validation_data)
    pr_model1[j == ii] <- ifelse(prob_sub > 0.5, 1, 0)
    
    # Model 2
    rf_full <- randomForest(Survived ~ ., data = train_data)
    pr_model2[j == ii] <- predict(object = rf_full, newdata = validation_data)
    
    # Model 3 (Python)
    pr_model3[j == ii] <- python_model3(data = train_data, newdata = validation_data) 
    
    # Alternatively, make a function that makes multiple models
    # pr_python_models will be a 2D array
    pr_python_models[j == ii, ] <- python_models(data = train_data, newdata = validation_data)
    
  }
  
  # TODO: Convert to list and use do.call rbind
  return(data.frame(index = i, 
                    model1 = with(train, mean(Survived == pr_lr_full)),
                    model2 = with(train, mean(Survived == pr_lr_sub)),
                    model3 = with(train, mean(Survived == pr_rf_full - 1))))
}
accuracy <- do.call(rbind, result)
