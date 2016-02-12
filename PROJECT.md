---
title: 'PRACTICAL MACHINE LEARNING: COURSE PROJECT'
author: "Pablo Morán Collantes"
date: "January-February 2016"
output: html_document
fontsize: 11pt

---

**BACKGROUND**

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

**REQUIRED PACKAGES**

To start with, we will load the required packages for the project.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Versión 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Escriba 'rattle()' para agitar, sacudir y  rotar sus datos.
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(e1071)
```

**DOWNLOAD REQUIRED DATA**

The downloaded data will be stored in the data frames train (training data) and test (test data).


```r
setInternet2(TRUE)
fileurl_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileurl_train, destfile = "./training_data.csv")
train <- read.csv("training_data.csv", na.strings=c("", "NA", "NULL"))
fileurl_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileurl_test, destfile = "./test_data.csv")
test <- read.csv("test_data.csv", na.strings=c("", "NA", "NULL"))
```

After checking the different variables in each data frame, we'll assure all the numeric variables have a numeric class assigned.


```r
train[,-c(1,2,160)]<-sapply(train[,-c(1,2,160)], as.numeric)
test[,-c(1,2,160)]<-sapply(test[,-c(1,2,160)], as.numeric)
```

**RANDOM SPLIT**

In order to perform sample analyses with the training data, we will split it into two different data frames (the first will contain 70% of data, the second will contain 30% of data).


```r
set.seed(1)
partition_Train <- createDataPartition(y=train$classe, p=0.7, list=F)
train1 <- train[partition_Train,]
train2 <- train[-partition_Train,]
```

**CHECK VARIABLES WITH EXTREMELY LOW VARIANCE**

Our main goal when we are interested in cleaning data is to remove those variables which do not have a clear importance in our analyses. In this case, we will remove those variables with extremely low variance. Afterwards, we will remove those containing NAs.


```r
zeroVAR <- nearZeroVar(train1, saveMetrics = TRUE)
train1 <- train1[,zeroVAR[, 'nzv']==0]
train1 <- train1[ , colSums(is.na(train1)) == 0]
train1 <- train1[,-1]
```

**MODIFY train2 AND test**


```r
var_train2 <- names(train1)
var_test <- names(train1[,-length(train1)])
train2 <- train2[,var_train2]
test <- test[,var_test]
```

**DATA COERCION**

In order to perform correctly the Machine Learning Algorithms, we have to make sure that the variables in our testing data frame are the same as in the created train1 df.


```r
for (i in 1:length(test) ) {
  for(j in 1:length(train1)) {
      if(length(grep(names(train1[i]), names(test)[j]))==1)  {
        class(test[j]) <- class(train1[i])
        }      
      }      
    }
```


**TYPE OF ANALYSIS: DECISION TREE**


```r
fit1 <- rpart(classe ~ ., data=train1, method="class")
fancyRpartPlot(fit1)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png)

This decision tree shows the different probabilities assigned for the classe variable in the train1 data frame. To obtain numeric values for the analysis, we will predict the model and test the results using confusionMatrix():


```r
pred1 <- predict(fit1, train2, type = "class")
confusionMatrix(pred1, train2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1566   34    1    8    8
##          B    6  906   47   18   34
##          C   53  195  938  124   47
##          D    0    0   31  772   32
##          E   49    4    9   42  961
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8739          
##                  95% CI : (0.8652, 0.8823)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8408          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9355   0.7954   0.9142   0.8008   0.8882
## Specificity            0.9879   0.9779   0.9138   0.9872   0.9783
## Pos Pred Value         0.9685   0.8961   0.6912   0.9246   0.9023
## Neg Pred Value         0.9747   0.9522   0.9806   0.9620   0.9749
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2661   0.1540   0.1594   0.1312   0.1633
## Detection Prevalence   0.2748   0.1718   0.2306   0.1419   0.1810
## Balanced Accuracy      0.9617   0.8867   0.9140   0.8940   0.9333
```

**TYPE OF ANALYSIS: RANDOM FOREST**


```r
fit2 <- randomForest(classe ~. , data = train1, method="class")
pred2 <- predict(fit2, train2, type = "class")
confusionMatrix(pred2, train2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    1    0    0    0
##          B    0 1138    2    0    0
##          C    0    0 1024    0    0
##          D    0    0    0  963    1
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9992         
##                  95% CI : (0.998, 0.9997)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9989         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9991   0.9981   0.9990   0.9991
## Specificity            0.9998   0.9996   1.0000   0.9998   0.9998
## Pos Pred Value         0.9994   0.9982   1.0000   0.9990   0.9991
## Neg Pred Value         1.0000   0.9998   0.9996   0.9998   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1934   0.1740   0.1636   0.1837
## Detection Prevalence   0.2846   0.1937   0.1740   0.1638   0.1839
## Balanced Accuracy      0.9999   0.9994   0.9990   0.9994   0.9994
```

As expected, the Random Forest analysis results show a higher accuracy that those obtained in the Decision Tree analysis.

**GENERATE FILES**


```r
pred2 <- predict(fit2, test, type = "class")
pred_cases = function(x){
  for(i in 1:length(x)){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pred_cases(pred2)
```
