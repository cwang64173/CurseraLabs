

library(caret)
library(corrplot)
library(corrplot)
library(data.table)
library(gbm)
library(ggplot2)
library(knitr)
library(lattice)
library(plotly)
library(randomForest)
library(rattle)
library(RColorBrewer)
library(rpart)
library(rpart.plot)

data_training <- read.csv("pml-training.csv")
data_quiz <- read.csv("pml-testing.csv")
dim(data_training)
dim(data_quiz)

# remove NA
in_training  <- createDataPartition(data_training$classe, p=0.70, list=FALSE)
training_set <- data_training[ in_training, ]
testing_set  <- data_training[-in_training, ]
dim(training_set)
dim(testing_set)

# remove NZV variables
nzv_var <- nearZeroVar(training_set)
training_set <- training_set[ , -nzv_var]
testing_set  <- testing_set [ , -nzv_var]
dim(training_set)


# remove mostly NA variables
na_var <- sapply(training_set, function(x) mean(is.na(x))) > 0.95
training_set <- training_set[ , na_var == FALSE]
testing_set  <- testing_set [ , na_var == FALSE]
dim(training_set)
dim(testing_set)

# remove columns 1 to 5 identification variables 
training_set <- training_set[ , -(1:5)]
testing_set  <- testing_set [ , -(1:5)]
dim(training_set)
dim(testing_set)

# correlation among variables
corr_matrix <- cor(training_set[ , -54])
corrplot(corr_matrix, order = "FPC", method = "circle", type = "lower",
         tl.cex = 0.6, tl.col = rgb(0, 0, 0))

# try Decision Tree model
set.seed(1967)
fit_DT <- rpart(classe ~ ., data = training_set, method="class")
fancyRpartPlot(fit_DT)

predict_DT <- predict(fit_DT, newdata = testing_set, type="class")
conf_matrix_DT <- confusionMatrix(table(predict_DT, testing_set$classe))
conf_matrix_DT

plot(conf_matrix_DT$table, col = conf_matrix_DT$byClass, 
     main = paste("Decision Tree Model: Predictive Accuracy =",
                  round(conf_matrix_DT$overall['Accuracy'], 4)))

# try Generalized Boosted Model 
set.seed(1967)
ctrl_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_GBM  <- train(classe ~ ., data = training_set, method = "gbm",
                  trControl = ctrl_GBM, verbose = FALSE)
fit_GBM$finalModel

predict_GBM <- predict(fit_GBM, newdata = testing_set)
conf_matrix_GBM <- confusionMatrix(table(predict_GBM, testing_set$classe))
conf_matrix_GBM


# try Random Forest
set.seed(1967)
ctrl_RF <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_RF  <- train(classe ~ ., data = training_set, method = "rf",
                 trControl = ctrl_RF, verbose = FALSE)
fit_RF$finalModel

predict_RF <- predict(fit_RF, newdata = testing_set)
conf_matrix_RF <- confusionMatrix(table(predict_RF, testing_set$classe))
conf_matrix_RF  

plot(fit_RF)

print(fit_RF$bestTune)

cat("Predictions: ", paste(predict(fit_RF, data_quiz)))










