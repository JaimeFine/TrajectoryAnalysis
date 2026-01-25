library(dplyr)
library(ggplot2)

data <- read.csv("alpha_sensitivity_report.csv")

zoi_s <- data$ZOI.numbers

ggplot(data, aes(x = alpha)) +
  geom_line(aes(y = zoi_s, color = "ZOI Points"), linewidth = 1) +
  geom_point(aes(y = zoi_s, color = "ZOI Points")) +
  scale_y_continuous(
    limits = c(37000, 41000),   # descending axis
    breaks = c(37000, 38000, 39000, 40000, 41000),
    name = "Number of ZOI Points"
  ) +
  labs(
    x = expression(alpha),
    y = "Number of ZOI Points"
  ) +
  theme_minimal()

data

