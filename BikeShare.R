library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)

## -------------------------
## 1) LOAD
## -------------------------
bike_train <- vroom("train.csv")
bike_test  <- vroom("test.csv")

## -------------------------
## 2) CLEANING (before modeling, train **only**)
##    - remove casual, registered
##    - change count to log(count)
## -------------------------
bike_train <- bike_train %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))   # NOTE: log-transform only on training per HW

## -------------------------
## 3) FEATURE ENGINEERING (recipe, before modeling)
##    Must do:
##      - recode weather 4 -> 3 then make factor
##      - extract hour from timestamp
##      - make season a factor
##      - one other step (here: dummy-encode all nominal predictors)
##    Also remove timestamp column after extracting hour.
## -------------------------
bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  # recode + factors
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(season  = factor(season)) %>%
  
  # hour → cyclic
  step_time(datetime, features = "hour") %>%
  step_mutate(
    hour_num = as.numeric(datetime_hour),
    hour_sin = sin(2 * pi * hour_num / 24),
    hour_cos = cos(2 * pi * hour_num / 24)
  ) %>%
  
  # season → cyclic
  step_mutate(
    season_sin = sin(2 * pi * as.numeric(season) / 4),
    season_cos = cos(2 * pi * as.numeric(season) / 4)
  ) %>%
  
  # clean up raw cols
  step_rm(datetime_hour, hour_num, datetime) %>%
  
  # one-hot the categorical (weather and any others)
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # --- INTERACTIONS ---
  # weather × hour (cyclic)
  step_interact( ~ (starts_with("weather_")):(hour_sin + hour_cos)) %>%
  # season × hour (cyclic × cyclic)
  step_interact( ~ (season_sin + season_cos):(hour_sin + hour_cos)) %>%
  # weather × season (dummies × cyclic)
  step_interact( ~ (starts_with("weather_")):(season_sin + season_cos)) %>%
  
  # finishers
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())
## Prep once so we can show the baked training rows later
prepped <- prep(bike_recipe)
baked_train <- bake(prepped, new_data = NULL)



## -------------------------
## Regression Tree
## -------------------------

tree_mod <- decision_tree(
  cost_complexity = tune(),
  tree_depth      = tune(),
  min_n           = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(tree_mod)

grid_of_tuning_params <- grid_regular(
  cost_complexity(),   # rpart's cp
  tree_depth(),        # max depth
  min_n(),             # min observations to split
  levels = 10
)

folds <- vfold_cv(bike_train, v = 10, strata = NULL)

CV_results <- tree_wf %>%
  tune_grid(
    resamples = folds,
    grid      = grid_of_tuning_params,
    metrics   = metric_set(rmse, mae)
  )

# Look at RMSE across hyperparameters (correct aesthetics)
library(ggplot2)
collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(aes(x = cost_complexity, y = mean, color = factor(tree_depth))) +
  geom_line() +
  geom_point() +
  scale_x_log10() +
  labs(color = "tree_depth")

bestTune <- CV_results %>% select_best(metric = "rmse")

final_fit <- tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train)

## -------------------------
## Predict on test + submission
## -------------------------
test_preds <- predict(final_fit, new_data = bike_test) %>%
  bind_cols(bike_test %>% select(datetime))

kaggle_submission <- test_preds %>%
  transmute(
    datetime = format(datetime, "%Y-%m-%d %H:%M:%S"),
    count    = pmax(0, round(exp(.pred)))  # back-transform, clamp, make integer
  )

file_name <- sprintf(
  "KagglePreds_tree_cp%s_depth%s_min%s.csv",
  formatC(bestTune$cost_complexity, format = "fg", digits = 6),
  bestTune$tree_depth,
  bestTune$min_n
)

vroom_write(kaggle_submission, file_name, delim = ",")

## Create a workflow with model & recipe

## Set up grid of tuning values

## Set up K-fold CV

## Find best tuning parameters

## Finalize workflow and predict




