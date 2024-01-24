library(rstan)
library(brms)
library(rstudioapi)
library(dplyr)
library(caret)
library(pROC)

train <- read.csv("D:/2023_Files/Study_202312/ABDA_Pre/train.csv")


df <- as.data.frame(train)
sample_size <- 5000
random_indices <- sample(nrow(df), sample_size)
data <- df[random_indices, ]


set.seed(666)  
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


# Bayesian logistic regression model with optimized settings
bayesian_model <- brm(
  formula = Response ~ Age + Gender + Annual_Premium + Policy_Sales_Channel + Vehicle_Age + Vehicle_Damage,
  data = train,
  family = bernoulli("logit"),
  prior = c(
      set_prior("normal(38.78, 15.80)", class = "b", coef = "Age"),
      set_prior("normal(0.55, 0.50)", class = "b", coef = "Gender"),
      set_prior("normal(30744.83, 16396.44)", class = "b", coef = "Annual_Premium"),
      set_prior("normal(111.72, 54.81)", class = "b", coef = "Policy_Sales_Channel"),
      set_prior("normal(0.60, 0.57)", class = "b", coef = "Vehicle_Age"),
      set_prior("normal(0.50, 0.50)", class = "b", coef = "Vehicle_Damage")
  ),
  chains = 2,
  iter = 4000,  
  warmup = 2000,  
  control = list(adapt_delta = 0.99, max_treedepth = 15),  # Adjusted control parameters
  thin = 10,  # Thinning the sample
  seed = 123
)

summary(bayesian_model)

# Predict probabilities
# Generate posterior predictions
posterior_predictions <- posterior_predict(bayesian_model, newdata = test,type = "Response")
predicted_probabilities <- apply(posterior_predictions, 2, mean)

# ROC curve to find the best threshold
roc_obj <- roc(test$Response, predicted_probabilities)
coords(roc_obj, "best", best.method = "closest.topleft")
optimal_threshold <- coords(roc_obj, "best", best.method = "closest.topleft")$threshold
binary_predictions <- ifelse(predicted_probabilities > optimal_threshold, 1, 0)
conf_matrix <- table(Predicted = binary_predictions, Actual = test$Response)


# Calculate the accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Precision: True Positives / (True Positives + False Positives)
precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
recall <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Output results
cat("Confusion Matrix:\n")
print(conf_matrix)
cat("Accuracy: ", accuracy, "\n")
cat("F1 Score: ", f1_score, "\n")

