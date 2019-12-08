##################################################################################################################
setwd("./Desktop/BCS_Term5_2019/CPSC340/project/new_data/processed/")
library(data.table)
library(LiblineaR)
library(tidyverse)
library(magrittr)
library(MLmetrics)
library(lubridate)
library(Rtsne)


# Read df
set.seed(42)
train <- fread("../../data/train.csv")

# df <- data.matrix(df)
keeps <- c("Dates", "Category", "DayOfWeek", "PdDistrict", "Resolution", "X", "Y")
cols <-  c("Category", "DayOfWeek", "PdDistrict", "Resolution")
train <- subset(train, select = keeps)

train %<>%
  mutate_each_(funs(factor(.)),cols)

train$year <- year(ymd_hms(train$Dates))
train$month <- month(ymd_hms(train$Dates))
train$day <- day(ymd_hms(train$Dates))
train$hour <- hour(ymd_hms(train$Dates))
train$minute <- minute(ymd_hms(train$Dates))
train$second <- second(ymd_hms(train$Dates))
train$Dates <- NULL
train[,11]
train <- data.matrix(train)
x <- train[,3:ncol(train)-1]
y <- train[,1]


# 3 hrs 
x <- scale(x)
tsne_out <- Rtsne(x[,c(2,9)], 
                  dims = 2, 
                  initial_dims = 3, 
                  perplexity = 2, 
                  theta = 0.35, 
                  check_duplicates = FALSE,
                  pca = FALSE, 
                  partial_pca = FALSE, 
                  max_iter = 100000, 
                  verbose = FALSE, 
                  is_distance = FALSE, 
                  Y_init = NULL, 
                  pca_center = TRUE, 
                  pca_scale = FALSE,
                  normalize = TRUE, 
                  momentum = 0.5, 
                  final_momentum = 0.8, 
                  eta = 200,
                  exaggeration_factor = 12, 
                  num_threads = 0)

plot(tsne_out$Y, col = y, asp=1)
xsaveRDS(tsne_out, "tsne01.rds")

# time and log loss plots
dat <- data.frame(model = c("Logistic SGD",
                            "Logistic SAG",
                            #"Modified Huber Loss SGD",
                            "Hinge SGD",
                            "Hinge CGD",
                            "Squared Hinge CGD",
                            "LDA",
                            "QDA",
                            "XGBoost",
                            "CatBoost",
                            "LightGBM",
                            "Random Forest"), 
                  testLogLoss = c(2.63452,
                                  2.51241,
                                  #3.62403,
                                  2.64329,
                                  2.66857,
                                  2.66145,
                                  2.53357,
                                  2.65619,
                                  2.24266,
                                  2.27116,
                                  2.27297,
                                  2.32384),
                  time = c(18.36,
                           72.44,
                           #18.39,
                           135.14,
                           1366.92,
                           2597.61,
                           3.81,
                           0.99,
                           20000.00,
                           10800.00,
                           5000.00,
                           283.55)
                  )

p<-ggplot(dat, aes(x=reorder(model, -testLogLoss), y=testLogLoss)) +
  # geom_bar(stat="identity")+
  theme_economist() +
  coord_flip() +
  geom_col(aes(fill = -log(time))) +
  xlab("Model") + 
  ylab("Test log loss") +
  ggtitle("Model accuracy vs. time") +
  scale_fill_gradient(name="Time") +
  theme(axis.text.y = element_text(size=18),
        axis.title.x = element_text(size=20),
        axis.title = element_text(size=20))
    
p
