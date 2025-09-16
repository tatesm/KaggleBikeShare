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
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(season = factor(season)) %>%
  step_time(datetime, features = "hour") %>%                 # creates `datetime_hour`
  step_mutate(
    hour_num = as.numeric(datetime_hour),                    # make numeric, avoid name clash
    hour_sin = sin(2 * pi * hour_num / 24),
    hour_cos = cos(2 * pi * hour_num / 24)
  ) %>%
  step_rm(datetime_hour, hour_num, datetime) %>%             # clean up
  step_mutate(
    season_sin = sin(2 * pi * as.numeric(season) / 4),
    season_cos = cos(2 * pi * as.numeric(season) / 4)
  ) %>%
  step_rm(season) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


## Prep once so we can show the baked training rows later
prepped <- prep(bike_recipe)
baked_train <- bake(prepped, new_data = NULL)


## -------------------------
## 4) PENALIZED REGRESSION, NORMAL
## -------------------------

preg_normodel <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

# Workflow with predictors/outcome
preg_normwf <- workflow() %>%
  add_model(preg_normodel) %>%
  add_recipe(bike_recipe)

# Define a grid of penalty/mixture combos
lambda_grid <- grid_regular(
  penalty(range = c(-3, 1)), # log10 penalty: 10^-3 to 10^1
  mixture(),                 # values from 0 to 1
  levels = 5                 # number of values per parameter
)

# Fit across resamples (e.g. cross-validation)
cv_folds <- vfold_cv(bike_train, v = 5)

tuned_res <- tune_grid(
  preg_normwf,
  resamples = cv_folds,
  grid = lambda_grid,
  control = control_grid(save_pred = TRUE)
)

collect_metrics(tuned_res)

top_5 <- show_best(tuned_res, metric = "rmse", n = 5)
top_5

all_preds <- collect_predictions(tuned_res)

# Keep only predictions where penalty/mixture match the top 5
top5_preds <- all_preds %>%
  semi_join(top_5, by = c("penalty", "mixture"))

for (i in 1:nrow(top_5)) {
  
  # Get the i-th best parameters
  params <- top_5[i, ]
  
  # Finalize workflow with these parameters
  final_wf <- finalize_workflow(preg_normwf, params)
  
  # Fit on full training data
  fit_wf <- fit(final_wf, data = bike_train)
  
  # Predict on test set (still log(count))
  test_preds <- predict(fit_wf, new_data = bike_test)
  
  # Build Kaggle submission
  kaggle_submission <- bike_test %>%
    select(datetime) %>%
    mutate(count = pmax(0, exp(test_preds$.pred))) %>%  # back-transform + clamp
    mutate(datetime = format(datetime))                 # "YYYY-MM-DD HH:MM:SS"
  
  # File name includes penalty + mixture for traceability
  file_name <- paste0(
    "KagglePreds_pen", round(params$penalty, 5),
    "_mix", round(params$mixture, 2),
    ".csv"
  )
  
  # Write CSV
  vroom_write(kaggle_submission, file_name, delim = ",")
}



