library(rstan)
library(brms)
library(rstudioapi)
library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(bayesplot)
library(gt)  
library(gtsummary)


train <- read.csv("train.csv")

set.seed(666)  
df <- as.data.frame(train)
sample_size <- 5000
random_indices <- sample(nrow(df), sample_size)
data <- df[random_indices, ]



indices <- createDataPartition(data$Response, p = 0.2, list = FALSE)
train <- data[-indices, ]
test <- data[indices, ]

# Transform 'Gender' column
train$Gender <- ifelse(train$Gender == 'Male', 1, 0)
test$Gender <- ifelse(test$Gender == 'Male', 1, 0)

# Transform 'Vehicle_Age' column
train$Vehicle_Age <- ifelse(train$Vehicle_Age == '> 2 Years', 2,
                            ifelse(train$Vehicle_Age == '1-2 Year', 1, 0))
test$Vehicle_Age <- ifelse(test$Vehicle_Age == '> 2 Years', 2,
                           ifelse(test$Vehicle_Age == '1-2 Year', 1, 0))

# Transform 'Vehicle_Damage' column
train$Vehicle_Damage <- ifelse(train$Vehicle_Damage == 'Yes', 1, 0)
test$Vehicle_Damage <- ifelse(test$Vehicle_Damage == 'Yes', 1, 0)

mean(train$Age)
sd(train$Age)

mean(train$Gender)
sd(train$Gender)

mean(train$Annual_Premium)
sd(train$Annual_Premium)

mean(train$Policy_Sales_Channel)
sd(train$Policy_Sales_Channel)

mean(train$Vehicle_Age)
sd(train$Vehicle_Age)

mean(train$Vehicle_Damage)
sd(train$Vehicle_Damage)