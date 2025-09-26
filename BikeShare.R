library(tidyverse)
library(tidymodels)
library(vroom)
library(bonsai)
library(lightgbm)
library(agua) 
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
##
##    Also remove timestamp column after extracting hour.
## -------------------------
bike_train <- bike_train %>% mutate(datetime = as.POSIXct(datetime))
bike_test  <- bike_test  %>% mutate(datetime = as.POSIXct(datetime))

bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(
    weather = ifelse(weather == 4, 3, weather),
    weather = factor(weather),
    season  = factor(season)
  ) %>%step_date(datetime, features = c("dow", "month", "year")) %>%
  step_time(datetime, features = c("hour")) %>%
  # Cyclical encoding for hour + month (useful for tree models, required for linear)
  step_mutate(
    # convert ordered factors to numeric *by value*, not by level index
    hour = as.integer(datetime_hour),
    month = as.integer(datetime_month)
  ) %>%
  
  step_mutate(
    hour_sin  = sin(2 * pi * hour/24),
    hour_cos  = cos(2 * pi * hour/24),
    month_sin = sin(2 * pi * month/12),
    month_cos = cos(2 * pi * month/12)
  ) %>%
  # Drop originals if you want
  step_rm(datetime, datetime_hour, datetime_month, hour, month) %>%
  step_zv(all_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%   # handle unseen levels in test
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

prepped <- prep(bike_recipe)
baked_train <- bake(prepped, new_data = NULL)

## Boosted Tree

boost_model <- boost_tree(  trees          = tune(),
                            tree_depth     = tune(),
                            learn_rate     = tune()
                            ) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("regression")



boost_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(boost_model)

boost_params <- parameters(
  trees(range = c(200, 1200)),
  learn_rate(range = c(-4, -1)),  # 10^-4 to 10^-1  (~0.0001 to 0.1)
  tree_depth(range = c(3L, 9L))
)

boost_grid <- grid_regular(boost_params, levels = 3)
## Set up K-fold CV
folds <- vfold_cv(bike_train, v = 6, strata = NULL)

## Find best tuning parameters
CV_results <- boost_wf %>%
  tune_grid(
    resamples = folds,
    grid      = boost_grid,
    metrics   = metric_set(rmse)
  )


collect_metrics(CV_results)

## Find best tuning parameters

bestTune <- CV_results %>% select_best(metric = "rmse")

final_fit <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train)

test_preds <- predict(final_fit, new_data = bike_test) %>%
  bind_cols(bike_test %>% select(datetime))

kaggle_submission <- test_preds %>%
  transmute(
    datetime = format(datetime, "%Y-%m-%d %H:%M:%S"),
    count    = pmax(0, round(exp(.pred)))  # back-transform, clamp, make integer
  )


file_name <- sprintf(
  "KagglePreds_lgbm_lr%0.3f_depth%d_trees%d.csv",
  bestTune$learn_rate,
  bestTune$tree_depth,
  bestTune$trees
)

vroom_write(kaggle_submission, file_name, delim = ",")





## Initialize an h2o session
h2o::h2o.init()

## Define the model
## max_runtime_secs = how long to let h2o.ai run
## max_models = how many models to stack
auto_model <- auto_ml() %>%
  set_engine("h2o", max_models=10) %>%
  set_mode("regression")

## Combine into Workflow
automl_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(auto_model) %>%
  fit(data=bike_train)

## Predict
preds <- predict(...)






