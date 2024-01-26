library(rstan)
library(brms)
library(rstudioapi)
library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(bayesplot)

train <- read.csv("train.csv")


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
  chains = 3,
  iter = 4000,  
  warmup = 2000,  
  control = list(adapt_delta = 0.99, max_treedepth = 15),  # Adjusted control parameters
  thin = 10,  # Thinning the sample
  seed = 123
)

#plotting of model

mcmc_plot(bayesian_model, type="trace")
mcmc_rank_ecdf(bayesian_model, plot_diff=TRUE)
mcmc_rank_overlay(bayesian_model)

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

# # Plot ROC curve
# plot(roc_obj, print.auc = TRUE, print.auc.col = "red", print.auc.adj = c(-1, 4), main = "ROC Curve")
# 
# 
# plot(test$Response, predicted_probabilities, main = "Predicted vs Real Values", xlab = "Real Values", ylab = "Predicted Values")


# `data` has columns `predicted_probabilities` for predicted probabilities and `Response` for actual binary outcomes

data=data_frame(predicted_probabilities,Response = test$Response)

data <- data %>%
  mutate(bin = cut(predicted_probabilities, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)) %>%
  group_by(bin) %>%
  summarize(mean_actual = mean(Response))

ggplot(data, aes(x = bin, y = mean_actual)) +
  geom_col() +
  theme_minimal() +
  labs(x = "Predicted Probability Bin", y = "Mean Actual Outcome", title = "Binned Residual Plot")


# Plot ROC curve
plot(roc_obj, main = "ROC Curve", col = "#1c61b6", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

# Add optimal threshold point
optimal_coords <- coords(roc_obj, "best", best.method = "closest.topleft")
points(optimal_coords[1], optimal_coords[2], pch = 19, col = "red", cex = 1.5)

# Adding labels and title
legend("bottomright", legend = c("ROC Curve", "Optimal Threshold"), 
       col = c("#1c61b6", "red"), lty = c(1, NA), pch = c(NA, 19))


calibration_data <- data.frame(
  Observed = test$Response,
  Predicted = predicted_probabilities
)

calibration_data <- calibration_data %>%
  mutate(Bin = cut(Predicted, breaks = seq(0, 1, length.out = 11), include.lowest = TRUE, labels = seq(0.1, 1, length.out = 10))) %>%
  group_by(Bin) %>%
  summarize(
    Mean_Predicted = mean(as.numeric(as.character(Bin))),
    Mean_Observed = mean(Observed)
  )

# Create the calibration plot
ggplot(calibration_data, aes(x = Mean_Predicted, y = Mean_Observed)) +
  geom_point() +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Mean Predicted Probability", y = "Mean Observed Outcome", title = "Calibration Plot") +
  xlim(0, 1) +
  ylim(0, 1)


#Bayesian_gam
bayesian_adjusted_model <- brm(
  formula = bf(
    Response ~ s(Age, k = 5) + Gender + s(Annual_Premium, k = 5) + Policy_Sales_Channel + Vehicle_Age + Vehicle_Damage,
    family = bernoulli("logit")
  ),
  data = train,
  prior = c(
    set_prior("normal(0, 1.5)", class = "b"),  # Priors for the fixed effect terms
    set_prior("student_t(3, 0, 2.5)", class = "sds", coef = "s(Age, k = 5)"),  # Prior for the smooth term of Age
    set_prior("student_t(3, 0, 2.5)", class = "sds", coef = "s(Annual_Premium, k = 5)")  # Prior for the smooth term of Annual_Premium
  ),
  chains = 3,
  iter = 4000,
  warmup = 2000,
  control = list(adapt_delta = 0.95, max_treedepth = 15),
  thin = 10,
  seed = 123
)

summary(bayesian_adjusted_model)
summary(bayesian_model)
# Predict probabilities
# Generate posterior predictions
posterior_predictions_m2 <- posterior_predict(bayesian_adjusted_model, newdata = test,type = "Response")               
predicted_probabilities_m2 <- apply(posterior_predictions_m2, 2, mean)

# ROC curve to find the best threshold
roc_obj_m2 <- roc(test$Response, predicted_probabilities_m2)
coords(roc_obj_m2, "best", best.method = "closest.topleft")
optimal_threshold_m2 <- coords(roc_obj_m2, "best", best.method = "closest.topleft")$threshold
binary_predictions_m2 <- ifelse(predicted_probabilities_m2 > optimal_threshold_m2, 1, 0)
conf_matrix_m2 <- table(Predicted = binary_predictions_m2, Actual = test$Response)


# Calculate the accuracy
accuracy_m2 <- sum(diag(conf_matrix_m2)) / sum(conf_matrix_m2)

# Precision: True Positives / (True Positives + False Positives)
precision_m2 <- conf_matrix_m2[2, 2] / sum(conf_matrix_m2[2, ])
recall_m2 <- conf_matrix_m2[2, 2] / sum(conf_matrix_m2[, 2])
f1_score_m2 <- 2 * (precision_m2 * recall_m2) / (precision_m2 + recall_m2)

# Output results
cat("Confusion Matrix:\n")
print(conf_matrix_m2)
cat("Accuracy_m2: ", accuracy_m2, "\n")
cat("F1 Score_m2: ", f1_score_m2, "\n")



