library(tidyverse)
library(tidymodels)
library(vroom)

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
  ) %>%
  step_time(datetime, features = "hour") %>%    # makes `datetime_hour`
  step_mutate(hour = as.numeric(datetime_hour)) %>%  # keep as `hour`
  step_rm(datetime_hour, datetime) %>%          # drop only source columns
  step_zv(all_predictors())

prepped <- prep(bike_recipe)
baked_train <- bake(prepped, new_data = NULL)

## Random Forest


## Create a workflow with model & recipe

randtree_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=tune()) %>% #Type of mode
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

randtree_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(randtree_mod)

## Set up grid of tuning values

# After have baked_train:
# baked_train <- bake(prepped, new_data = NULL)

rf_params <- parameters(
  # finalize mtry to the number of predictors after the recipe
  finalize(mtry(range = c(2L, ncol(baked_train) - 1L)), baked_train),
  min_n(range = c(5L, 20L)),
  trees(range = c(500L, 1500L))
)

rf_grid <- grid_regular(rf_params, levels = 3)
## Set up K-fold CV
folds <- vfold_cv(bike_train, v = 10, strata = NULL)

## Find best tuning parameters
CV_results <- randtree_wf %>%
  tune_grid(
    resamples = folds,
    grid      = rf_grid,
    metrics   = metric_set(rmse, mae)
  )


collect_metrics(CV_results)

## Find best tuning parameters

bestTune <- CV_results %>% select_best(metric = "rmse")

final_fit <- randtree_wf %>%
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
  "KagglePreds_random_forest_mtry%s_minn%s_trees%s.csv",
  formatC(bestTune$mtry, format = "fg", digits = 6),
  bestTune$min_n,
  bestTune$trees
)

vroom_write(kaggle_submission, file_name, delim = ",")









