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

## Initialize an h2o session
h2o::h2o.init()

## Define the model
## max_runtime_secs = how long to let h2o.ai run
## max_models = how many models to stack
auto_model <- auto_ml() %>%
  set_engine("h2o", max_models=10) %>%
  set_mode("regression")

automl_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(auto_model) %>%
  fit(data = bike_train)

preds <- predict(automl_wf, new_data = bike_test)  # tibble with .pred

# Extract the raw H2O AutoML object
aml <- extract_fit_engine(automl_wf)

# ✅ Leaderboard
lb <- aml@leaderboard
lb_df <- as.data.frame(lb)
print(head(lb_df))   # safer than indexing columns
# or View(lb_df)

# ✅ Get the leader model directly
leader <- aml@leader
print(leader@model_id)

# ✅ Inspect the leader model's parameters
h2o::h2o.getModel(leader@model_id)@allparameters








kaggle_submission <- preds %>%
  bind_cols(bike_test) %>%                        # add datetime column
  select(datetime, .pred) %>%                     # keep only datetime + predictions
  rename(count = .pred) %>%                       # rename prediction column to count
  mutate(count = pmax(0, round(exp(count)))) %>%  # back-transform, clamp to 0
  mutate(datetime = as.character(format(datetime)))  # format for Kaggle

## Write to CSV with comma delimiter
vroom::vroom_write(
  x = kaggle_submission,
  file = "./KagglePreds.csv",
  delim = ","
)



