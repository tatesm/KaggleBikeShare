library(tidyverse)
library(tidymodels)
library(patchwork)
library(vroom)


bike_train <- vroom("train.csv")
glimpse(bike_train)
# The important in the data set are probably weather, workingday, temp,and humidity as I believe these would have a major impact on count
summary(bike_train)

bike_train <- bike_train %>%
  mutate(workingday = factor(workingday, labels = c("Non-working", "Working")))

workingplot <- ggplot(data = bike_train, aes(x= workingday, y = count))+
  geom_boxplot()+
  labs(title = "Bike Count Distribution on Working vs Non Working Days",
       x = "Working Day")

weatherplot <- ggplot(data = bike_train, aes(x= weather))+
  geom_bar()+
  labs(title = "Weather", x = "Weather Type")

tempplot <- ggplot(data = bike_train, aes(x = temp, y = count))+
  geom_point(alpha = 0.3)+
  geom_smooth(se = F)

humidplot <- ggplot(data = bike_train, aes(x = humidity, y = count))+
  geom_point(alpha = 0.3)+
  geom_smooth(se = F)


(weatherplot + workingplot)/ (tempplot + humidplot)


#Linear Regression Model
bike_train <- vroom("train.csv")
testData   <- vroom("test.csv")

cat_cols <- c("season", "holiday", "workingday", "weather")
bike_train <- bike_train %>%
  mutate(across(all_of(cat_cols), factor))

testData <- testData %>%
  mutate(across(all_of(cat_cols),
                ~ factor(.x, levels = levels(bike_train[[cur_column()]]))))

bike_train <- bike_train %>%
  mutate(log_count = log(count))

my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

fitted_model <- my_linear_model %>%
  fit(log_count ~ season + holiday + workingday + weather +
        atemp + humidity + windspeed,
      data = bike_train)


bike_predictions <- predict(fitted_model, new_data = testData)  # .pred is log(count)

kaggle_submission <- bike_predictions %>%
  bind_cols(testData) %>%
  transmute(
    datetime = format(datetime),          # "YYYY-MM-DD HH:MM:SS"
    count    = pmax(0, exp(.pred))        # back-transform from log(count)
    
  )


vroom_write(x = kaggle_submission, path = "LinearPreds.csv", delim = ",")
