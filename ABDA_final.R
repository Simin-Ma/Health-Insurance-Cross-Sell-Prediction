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


# z-score 

continuous_vars <- c("Age", "Gender", "Annual_Premium", "Policy_Sales_Channel", "Vehicle_Age", "Vehicle_Damage")
train_standardized <- train

for(var in continuous_vars) {
  train_standardized[[var]] <- scale(train[[var]])
}

mean_values <- sapply(train_standardized[continuous_vars], mean)
sd_values <- sapply(train_standardized[continuous_vars], sd)

test_standardized <- test

for(var in continuous_vars) {
  test_standardized[[var]] <- scale(test[[var]])
}

# mean_values <- sapply(test_standardized[continuous_vars], mean)
# sd_values <- sapply(test_standardized[continuous_vars], sd)
# Create tables for each variable



summary_table <- train_standardized %>%
  summarise(across(
    .cols = c(Age, Gender, Region_Code, Annual_Premium, Policy_Sales_Channel, Vehicle_Age, Vehicle_Damage, Response),
    .fns = list(
      count = ~ n(),
      mean = ~ mean(.x, na.rm = TRUE),
      std = ~ sd(.x, na.rm = TRUE),
      min = ~ min(.x, na.rm = TRUE),
      `25%` = ~ quantile(.x, probs = 0.25, na.rm = TRUE),
      median = ~ median(.x, na.rm = TRUE),
      `75%` = ~ quantile(.x, probs = 0.75, na.rm = TRUE),
      max = ~ max(.x, na.rm = TRUE)
    )
  ))


# Bayesian logistic regression model with optimized settings
bayesian_model <- brm(
  formula = Response ~ Age + Gender + Annual_Premium + Policy_Sales_Channel + Vehicle_Age + Vehicle_Damage,
  data = train_standardized,
  family = bernoulli("logit"),
  prior = c(
    set_prior("normal(0, 1)", class = "b", coef = "Age"),
    set_prior("normal(0, 1)", class = "b", coef = "Gender"),
    set_prior("normal(0, 1)", class = "b", coef = "Annual_Premium"),
    set_prior("normal(0, 10)", class = "b", coef = "Policy_Sales_Channel"),
    set_prior("normal(0, 1)", class = "b", coef = "Vehicle_Age"),
    set_prior("normal(0, 1)", class = "b", coef = "Vehicle_Damage")
  ),
  chains = 4,
  iter = 4000,  
  warmup = 1000,  
  control = list(adapt_delta = 0.95),  
  seed = 123
)

summary(bayesian_model)

#construct second model
bayesian_gam <- brm(
  formula = Response ~ s(Age) + Gender + s(Annual_Premium) + Policy_Sales_Channel + Vehicle_Age + Vehicle_Damage,
  data = train,
  family = bernoulli("logit"),
  prior = c(
    set_prior("normal(0, 1)", class = "Intercept"),  # Assuming a baseline log-odds close to 0
    set_prior("normal(0, 2.5)", class = "b"),  # Default for fixed effects
    set_prior("student_t(3, 0, 2.5)", class = "sds")  # Regularizing smooth terms
  ),
  chains = 4,
  iter = 4000,
  warmup =1000,
  control = list(adapt_delta = 0.95),  # Adjusted control parameters
  seed = 123
)

summary(bayesian_model)

#posterior
# Predict probabilities
# Generate posterior predictions

mcmc_plot(bayesian_model, type="trace")
mcmc_rank_ecdf(bayesian_model, plot_diff=TRUE)
mcmc_rank_overlay(bayesian_model)

mcmc_plot(bayesian_gam, type="trace")
mcmc_rank_ecdf(bayesian_gam, plot_diff=TRUE)
mcmc_rank_overlay(bayesian_gam)


posterior_predictions <- posterior_predict(bayesian_model, newdata = test_standardized, type = "Response")
predicted_probabilities <- apply(posterior_predictions, 2, mean)

# ROC curve to find the best threshold
roc_obj <- roc(test_standardized$Response, predicted_probabilities)
coords(roc_obj, "best", best.method = "closest.topleft")
optimal_threshold <- coords(roc_obj, "best", best.method = "closest.topleft")$threshold
binary_predictions <- ifelse(predicted_probabilities > optimal_threshold, 1, 0)
conf_matrix <- table(Predicted = binary_predictions, Actual = test_standardized$Response)

# Calculate the accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Precision: True Positives / (True Positives + False Positives)
precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
# Recall: True Positives / (True Positives + False Negatives)
recall <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
# F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
f1_score <- 2 * (precision * recall) / (precision + recall)

# Output results
cat("Confusion Matrix:\n")
print(conf_matrix)
cat("Accuracy: ", accuracy, "\n")
cat("F1 Score: ", f1_score, "\n")

#2nd model
posterior_predictions1 <- posterior_predict(bayesian_gam, newdata = test_standardized, type = "Response")
predicted_probabilities1 <- apply(posterior_predictions1, 2, mean)

# ROC curve to find the best threshold
roc_obj1 <- roc(test_standardized$Response, predicted_probabilities1)
coords(roc_obj, "best", best.method = "closest.topleft")
optimal_threshold <- coords(roc_obj, "best", best.method = "closest.topleft")$threshold
binary_predictions <- ifelse(predicted_probabilities > optimal_threshold, 1, 0)
conf_matrix <- table(Predicted = binary_predictions, Actual = test_standardized$Response)

# Calculate the accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Precision: True Positives / (True Positives + False Positives)
precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
# Recall: True Positives / (True Positives + False Negatives)
recall <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
# F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
f1_score <- 2 * (precision * recall) / (precision + recall)

# Output results
cat("Confusion Matrix:\n")
print(conf_matrix)
cat("Accuracy: ", accuracy, "\n")
cat("F1 Score: ", f1_score, "\n")


pp_check(bayesian_gam)  # shows dens_overlay plot by default
pp_check(bayesian_model)
