install.packages(c('neuralnet','keras','tensorflow'),dependencies = T)

library(tidyverse)
library(neuralnet)


# Simulated Data ----------------------------------------------------------

Data <- tibble(Group = rep(c("A", "B", "C"), each = 100),
               X1 = c(rnorm(100, mean = 10, sd = 2), rnorm(100, mean = 15, sd = 2), rnorm(100, mean = 5, sd = 2)),
               X2 = c(rnorm(100, 2000, sd = 100), rnorm(100, 2500, sd = 100), rnorm(100, 3000, sd = 100)),
               X3 = c(rnorm(100, 5, sd = 0.5), rnorm(100, 9, sd = 0.5), rnorm(100, 7, sd = 0.5)))



ggplot(Data, aes(x = Group, y = X1)) +
  geom_boxplot()
ggplot(Data, aes(x = Group, y = X2)) +
  geom_boxplot()
ggplot(Data, aes(x = Group, y = X3)) +
  geom_boxplot()


# Data from R -------------------------------------------------------------

data("diamonds")

# Fitting Model w Real Data -----------------------------------------------

diamonds <- diamonds %>%
  mutate(cut = factor(cut))


# Breaking the data up allows for us to test the accuracy of the model on points that weren't used to build the network
Train_Rows <- sample(1:nrow(diamonds), size = floor(2*nrow(diamonds)/3))

Train <- diamonds[Train_Rows,]
Test <- diamonds[-Train_Rows,]


# For the model, we can use a typical formula that we would use in Regression. It is important to note that the predictors must be numeric (which is why PCA may be helpful here) and the response can be either, but the linear.output argument must be T if the response is numeric, and F otherwise. The "hidden = ___" argument is indicative of how many nodes we want in the hidden layer. In the example below, we want 4 nodes in the first layer and 2 in the second, hence "hidden = c(4,2)"  
model = neuralnet(cut ~ depth + table + price + x + y + z, data = Train, hidden = c(4, 2), linear.output = F) 

plot(model, rep = "best")

pred <- predict(model, Test)
labels <- c(levels(diamonds$cut))
prediction_label <- data.frame(max.col(pred)) %>%     
  mutate(pred=labels[max.col.pred.]) %>%
  select(2) %>%
  unlist()

table(Test$Group, prediction_label)

check = as.numeric(Test$Group) == max.col(pred)
accuracy = (sum(check)/nrow(Test))*100
print(accuracy)


# Fitting Model w PCA -----------------------------------------------------

prin.comps <- prcomp(scale(diamonds[Train_Rows,5:10]))

summary(prin.comps)
summary(prin.comps)$rotation

Train <- Train %>% bind_cols(prin.comps$x[,1:3])

model = neuralnet(cut ~ PC1 + PC2 + PC3, data = Train, hidden = c(4, 2), linear.output = F) 

plot(model, rep = "best")

prin.comps.test <- prcomp(scale(diamonds[-Train_Rows,5:10]))

Test <- Test %>% bind_cols(prin.comps.test$x[,1:3])

pred <- predict(model, Test)
labels <- c(levels(diamonds$cut))
prediction_label <- data.frame(max.col(pred)) %>%     
  mutate(pred=labels[max.col.pred.]) %>%
  select(2) %>%
  unlist()

table(Test$Group, prediction_label)

check = as.numeric(Test$Group) == max.col(pred)
accuracy = (sum(check)/nrow(Test))*100
print(accuracy)
