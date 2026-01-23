library(ggplot2)
library(dplyr)

data <- read.csv("sigma_sensitivity_report.csv")

ggplot(data, aes(x = sigma, y = num_communities)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_point(color = "darkgreen", size = 2) +
  labs(
    x = expression(sigma),
    y = "Number of Communities"
  ) +
  theme_minimal()

data