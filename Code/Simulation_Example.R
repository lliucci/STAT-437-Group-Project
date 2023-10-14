#install.packages(c('neuralnet','keras','tensorflow'),dependencies = T)

library(tidyverse)
library(neuralnet)
library(tictoc)
#devtools::install_github("agenis/GuessCompx")
library(GuessCompx)
#remotes::install_github("rstudio/tensorflow")
library(tensorflow)
#tf$config$list_physical_devices("GPU")

# AirQuality Data (Single Layer Example) ----------------------------------

data("airquality")
airquality <- airquality %>% drop_na()
prin.comps <- prcomp(scale(airquality[,2:6]))
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
airquality[,1:6] <- normalize(airquality[,2:6])
summary(prin.comps)
airquality <- airquality %>% bind_cols(prin.comps$x[,1:3])
Train_Rows <- sample(1:nrow(airquality), size = floor(2*nrow(airquality)/3)) # Breaking the data up allows for us to test the accuracy of the model on points that weren't used to build the network
Train <- airquality[Train_Rows,]
Test <- airquality[-Train_Rows,]

## Fitting Model with Data -------------------------------------------------

# For the model, we can use a typical formula that we would use in Regression. It is important to note that the predictors must be numeric (which is why PCA may be helpful here) and the response can be either, but the linear.output argument must be T if the response is numeric, and F otherwise. The "hidden = ___" argument is indicative of how many nodes we want in the hidden layer. In the example below, we want 4 nodes in the first layer and 2 in the second, hence "hidden = c(4,2)"  
tic()
model = neuralnet(Ozone ~ Solar.R + Wind + Temp + Month + Day, data = Train, hidden = 3, linear.output = T) 
time_to_fit_w_Data <- toc()


## Fitting Model w PCA -----------------------------------------------------

plot(prin.comps)
summary(prin.comps)
summary(prin.comps)$rotation

tic()
model.pc = neuralnet(Ozone ~ PC1 + PC2 + PC3, data = Train, hidden = 3, linear.output = T) 
ttf_w_PC <- toc()

## Assessing Model Accuracy ------------------------------------------------

### Testing Model trained on Data  

plot(model, rep = "best")
pred <- predict(model, Test)
mean(((Test$Ozone - pred)/Test$Ozone) <= 0.05) # Percent of predictions that are within 5% of observed value

### Testing Model trained on PCs

plot(model.pc, rep = "best")
pred.pc <- predict(model.pc, Test)
mean(((Test$Ozone - pred.pc)/Test$Ozone) <= 0.05) # Percent of predictions that are within 5% of observed value



# Iris Data (Multilayer Example) ------------------------------------------
data('iris')
prin.comps2 <- prcomp(scale(iris[,1:4]))
iris[,1:4] <- normalize(iris[,1:4])
iris <- bind_cols(iris, prin.comps2$x[,1:3])
Train_Rows2 <- sample(1:nrow(iris), size = floor(2*nrow(iris)/3))
Train2 <- iris[Train_Rows2,]
Test2 <- iris[-Train_Rows2,]

## With Data ---------------------------------------------------------------

tic()
model2 <- neuralnet(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = Train2, hidden = c(4,2), linear.output = F)
ttf_data_iris <- toc()

plot(model2.pc, rep = "best")

pred2 <- predict(model2, Test2)
labels <- c(levels(iris$Species))
prediction_label <- data.frame(max.col(pred2)) %>%     
  mutate(pred=labels[max.col.pred2.]) %>%
  select(2) %>%
  unlist()

table(Test2$Species, prediction_label)

check = as.numeric(Test2$Species) == max.col(pred2)
accuracy = (sum(check)/nrow(Test2))*100
print(accuracy)


## With PCA ----------------------------------------------------------------

summary(prin.comps2)

tic()
model2.pc <- neuralnet(Species ~ PC1 + PC2, data = Train2, hidden = c(4,2), linear.output = F)
ttf_pc_iris <- toc()

plot(model2.pc, rep = "best")

pred2.pc <- predict(model2.pc, Test2)
labels <- c(levels(iris$Species))
prediction_label <- data.frame(max.col(pred2)) %>%     
  mutate(pred=labels[max.col.pred2.]) %>%
  select(2) %>%
  unlist()

table(Test2$Species, prediction_label)

check = as.numeric(Test2$Species) == max.col(pred2)
accuracy = (sum(check)/nrow(Test2))*100
print(accuracy)

# Saving env --------------------------------------------------------------

save.image("437-single-layer-neural-nets.Rdata")
