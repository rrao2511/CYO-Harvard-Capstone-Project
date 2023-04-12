##2. Data Preparation and Data Preprocessing


## Dataset Source:

## This particular dataset - Boston housing has been downloaded from Kaggle. 
## Since Kaggle does not allow us to download the files directly , have downloaded the file to my github and here is the link to the file:

##https://github.com/rrao2511/CYO-Harvard-Capstone-Project/raw/main/housing.csv

## First step - download the Packages needed for this analysis

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

##Load all the libraries:

library(tidyverse)
library(ggplot2)
library(caret)
library(dplyr)
library(corrplot)
library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)


##Download the dataset 

## Since Kaggle does not allow us to download the files directly , have downloaded the file to my github and here is the link to the file:

##https://github.com/rrao2511/CYO-Harvard-Capstone-Project/raw/main/housing.csv


##Reading the data from the csv file

boston_housing<-read.csv("https://github.com/rrao2511/CYO-Harvard-Capstone-Project/raw/main/housing.csv",header=TRUE,sep=",",quote ="\"")



## for the purpose of this analysis we are looking at a subset of the Boston housing set

## First lets look at the data set - checking the dimension.
## This dataset has 489 observations and 4 columns. This is a subset of the original Kaggle dataset.

dim(boston_housing)

### structure of the data set

str(boston_housing)

## There are 4 columns and the details of the column are shown below.We will be using all the 4 columns for our analysis.

## Explanation of Column names and details 

## RM - Average number of rooms per dwelling

## LSTAT - % lower status of population

## PT Ratio - Pupil teacher ratio by town

## MEDV - Median Value of owner occupied homes in $1000s.


## Checking first few rows 

head(boston_housing)

##summary statistics

summary(boston_housing)

##Cleaning up the data 

## Since this dataset is already clean, data cleaning was not needed and it could be used directly for analysis.

## Check to see if there are duplicate values:

sum(duplicated(boston_housing))

##Check to see if there are missing values

sum(is.na(boston_housing))


## 3. Exploratory Data Analysis using Data Visualization

## Before we start building the model we will understand the data set by doing some Exploratory Data Analysis.

## Check the correlation between variables by plotting a correlation graph


corrplot(cor(boston_housing), method = "number", type = "upper", diag = FALSE)

## From correlation matrix, we observe that:

## Both RM and LSTAT have a strong correlation with MEDV. 

## Median value of owner-occupied homes (in 1000$) increases as average number of rooms per dwelling increases and it decreases if percent of lower status population in the area increases.

## PT Ratio has a positive correlation with LSTAT


## Now lets look at Scatter plots to show relationship between Median value and variables


boston_housing%>%
  gather(key, val,-MEDV) %>%
  ggplot(aes(x = val, y = MEDV/1000))+
  geom_point()+
  stat_smooth(method = "lm", se = TRUE, col ="blue") +
  facet_wrap(~key, scales = "free")+
  theme_grey()+
  ggtitle("Scatter plot - Dependent variables vs Median value(medv)")


## Observations:

## From the above plots we see that RM and LSTAT have a strong correlation with Median value.

## The Median value prices increases as the RM value increases linearly.

## The Median value prices tend to decrease with an increase 



## 4. Developing the Models

## We will use three different models for this project:

## Decision trees, Random Forest and Support Vector Machine.

## We will evaluate the models using Root Mean Squared Error (RMSE).

### First we need to split the data into train sets and test sets:

## Data is split into train and test sets - 80:20 

set.seed(123)
bh_index<- sample(nrow(boston_housing),nrow(boston_housing)*.80)
bh_train<- boston_housing[bh_index,]
bh_test<- boston_housing[-bh_index,]

##   Model Development

##Model 1 - Decision trees


##Create the model using Decision trees

bhtree.fit<- rpart(MEDV~., data= bh_train)


#Plotting the tree

rpart.plot(bhtree.fit, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE)


## Predict on the test set

tree.pred<- predict(bhtree.fit, newdata = bh_test)

##Calculate the RMSE:

tree.rmse<- sqrt(mean((bh_test$MEDV- tree.pred)^2))

##Print the RMSE

cat("Decision Tree RMSE", round(tree.rmse,2),"\n")




##Model 2 - Random Forest 


##Next we build a model using Random Forest on the training set:

rf.fit<- randomForest(MEDV~., data= bh_train, ntree= 500, mtry = 3)

##Predict on the test set

rf.pred<- predict(rf.fit, newdata = bh_test)


## Calculate the RMSE

rf.rmse<- sqrt(mean((bh_test$MEDV - rf.pred)^2))


## Print the rmse

cat("Random Forest RMSE", round(rf.rmse,2), "\n")



## Model 3 - Building the model using SVM 


svm.fit<- svm(MEDV~., data = bh_train, kernel= "linear", cost =1)

## Make prediction using the test set

svm.pred<- predict(svm.fit, newdata = bh_test)

## Calculate the Root mean squared error (RMSE) of the predictions

svm.rmse<- sqrt(mean((svm.pred- bh_test$MEDV)^2))


## Print the RMSE 

cat("SVM RMSE:", svm.rmse, "\n")

##################################################################

##  MODEL RESULTS

##################################################################


## 5. Conclusion  


## Create a table for the RMSE values of Decision trees, Random Forest and SVM

results_table<- data.frame(Model = c("Decision Tree", "Random Forest", "SVM"),
                           RMSE= c(tree.rmse,rf.rmse,svm.rmse ))


print(results_table)

## Based on the above results here are our observations:

##  a) The random forest model has the lowest RMSE value indicating that 
##     it may be the best model for predicting the median value of owner-occupied homes in Boston.

##  b) Whereas the decision tree model and SVM models have  higher RMSE values , 
##     indicating that they may be less accurate than the decision tree model.



## 6. Limitations of the Model

##  We need to be cautious we need to be cautious when drawing conclusions based on RMSE values alone, as there may be other factors to consider such as model complexity, 
##  interpretability, and computational efficiency

##  Random forest models can be further improved with hyperparameters tuning.
##  But on account of shortage of time this was not attempted. 

##  Similarly the SVM model could be tuned further by changing the parameters and the kernel function.
##  But on account of shortage of time this was not attempted. 




