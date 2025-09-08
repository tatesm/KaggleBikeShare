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
