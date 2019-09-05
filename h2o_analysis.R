library(h2o)
#start cmd prompt h2o cluster
h2o.init()

library(tidyverse)

h2o <- read_csv("h2o.csv", col_types = cols(Sample.age.conditions = col_factor(levels = c("1", "2", "3"))))

data <- as.h2o(h2o)
split <- h2o.splitFrame(data, ratios = 0.8)
train <- split[[1]]
test <- split[[2]]

features <- colnames(h2o)
features <- features[-1]
features <- features[1:1000]
target <- "Sample.age.conditions"

model <- h2o.deeplearning(x = features, y = target, training_frame = train, model_id = "deeplearning", nfolds = 5)

performance <- h2o.performance(model, newdata = test)

automl = h2o.automl(x = features, y = target, training_frame = train, nfolds = 5, max_models = 20, stopping_metric = "RMSE", project_name = "automl")

as.data.frame(automl@leaderboard)

performance_auto <- h2o.performance(automl@leader, newdata = test)

best_model <- automl@leader

h2o.saveModel(best_model, path = "best_model")