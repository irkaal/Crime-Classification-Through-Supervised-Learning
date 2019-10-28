# Packages
library(tidyverse)
library(styler)

# Styler
style_file("San_francisco_crime_classification.R")

# Data
train <- read.csv("./data/train.csv")
train$Id <- NA
test <- read.csv("./data/test.csv")
test <- test %>%
  mutate(
    Category = NA,
    Descript = NA,
    Resolution = NA
  )
crime_data <- rbind(train, test)

#
# Feature Engineering:
#

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


# 4. Encode DayOfWeek into dummy variables
crime_data <- crime_data %>%
  mutate(
    Monday = ifelse(DayOfWeek == "Monday", 1, 0),
    Tuesday = ifelse(DayOfWeek == "Tuesday", 1, 0),
    Wednesday = ifelse(DayOfWeek == "Wednesday", 1, 0),
    Thursday = ifelse(DayOfWeek == "Thursday", 1, 0),
    Friday = ifelse(DayOfWeek == "Friday", 1, 0),
    Saturday = ifelse(DayOfWeek == "Saturday", 1, 0),
    Sunday = ifelse(DayOfWeek == "Sunday", 1, 0),
    DayOfWeek = NULL
  )


# 5. PdDistrict ~ PatrolDivision
# Patrol divisions are broken down into two divisions Golden Gate Division and Metro Division which are each led by San Francisco Police Commanders.
# https://en.wikipedia.org/wiki/San_Francisco_Police_Department
metro_div <- c("CENTRAL", "INGLESIDE", "NORTHERN", "SOUTHERN", "TENDERLOIN")
golden_gate_div <- c("BAYVIEW", "MISSION", "PARK", "RICHMOND", "TARAVAL")
crime_data <- crime_data %>%
  mutate(PatrolDivision = case_when(
    PdDistrict %in% metro_div ~ "Metro",
    PdDistrict %in% golden_gate_div ~ "GoldenGate"
  ))
