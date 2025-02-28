library(Metrics)

# Load preprocessed data
train_x <- fread("./project/volume/data/interim/train_x.csv")
test_x <- fread("./project/volume/data/interim/test_x.csv")
train_y <- fread("./project/volume/data/interim/train_y.csv")
test_id <- fread("./project/volume/data/interim/test_id.csv")

# Reattach response variable
data_train <- data.table(train_x)
data_train$result <- train_y$result

data_test <- data.table(test_x)

# Fit logistic regression model
model <- glm(result ~ ., data = data_train, family = "binomial")

# Model summary
summary(model)

# Save model
saveRDS(model, "./project/volume/models/coin_flip_model.rds")

# Predict probabilities on test set
test_probs <- predict(model, newdata = data_test, type = "response")

# Create submission file
submission <- data.frame(id = test_id$id, result = test_probs)
write.csv(submission, "./project/volume/data/processed/submission_coin_flip.csv", row.names = FALSE)
