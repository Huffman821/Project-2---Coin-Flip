library(data.table)
library(tidyverse)
library(caret)

set.seed(77)

# Load datasets
train <- fread("./project/volume/data/raw/train_file.csv")
test <- fread("./project/volume/data/raw/test_file.csv")

# Inspect structure
str(train)
str(test)

# Ensure `result` is a factor for classification
train$result <- as.factor(train$result)

# Keep 'id' for merging but exclude from modeling
train_id <- train$id
test_id <- test$id

train_x <- train %>% select(-id, -result)
test_x <- test %>% select(-id)
train_y <- train$result

# Apply dummy variable transformation (if needed)
dummies <- dummyVars(~ ., data = train_x)
train_x <- predict(dummies, newdata = train_x) %>% data.table()
test_x <- predict(dummies, newdata = test_x) %>% data.table()

# Save preprocessed data
fwrite(train_x, "./project/volume/data/interim/train_x.csv")
fwrite(test_x, "./project/volume/data/interim/test_x.csv")
fwrite(data.table(id = train_id, result = train_y), "./project/volume/data/interim/train_y.csv")
fwrite(data.table(id = test_id), "./project/volume/data/interim/test_id.csv")

saveRDS(dummies, "./project/volume/models/coin_flip_dummies.rds")
