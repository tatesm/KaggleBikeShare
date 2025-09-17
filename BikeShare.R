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
## 4) PENALIZED REGRESSION, NORMAL
## -------------------------

preg_normodel <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

# Workflow with predictors/outcome
preg_normwf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_normodel)

grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5)

# Fit across resamples (e.g. cross-validation)
folds <- vfold_cv(bike_train, v = 5)

CV_results <- preg_normwf %>%
tune_grid(resamples=folds,
          grid=grid_of_tuning_params,
          metrics=metric_set(rmse, mae)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()


## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best(metric="rmse")

final_wf <-
  preg_normwf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bike_train)

## Predict
final_wf %>%
  predict(new_data = bike_test)
  
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




## -------------------------
## 5) Optimal Model
## -------------------------



