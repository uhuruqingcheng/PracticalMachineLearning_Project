# Random Forest Approach Based Personal Activiy Recognition
By Qingcheng  
Saturday, August 23, 2014  

#### Couresra-[Practical Machine Learning][predmachlearn-004][Course Project 1]

# A. Synopsis
In this analysis document, we will apply **Random Forest Classification** algorithm to predict **personal activiy**. The data collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, which can be download:

> * [Training data][1].
> * [Test data][2].

The data for this project come from [Human Activity Recognition][3]. Besides of this website, more information can also be obtained in the paper [Qualitative Activity Recognition of Weight Lifting Exercises][4].


# B. Model Fitting

### B.1. Locale and environment


```r
Sys.setlocale('LC_ALL', 'English')
```

```
## [1] "LC_COLLATE=English_United States.1252;LC_CTYPE=English_United States.1252;LC_MONETARY=English_United States.1252;LC_NUMERIC=C;LC_TIME=English_United States.1252"
```

```r
sessionInfo()
```

```
## R version 3.1.1 (2014-07-10)
## Platform: i386-w64-mingw32/i386 (32-bit)
## 
## locale:
## [1] LC_COLLATE=English_United States.1252 
## [2] LC_CTYPE=English_United States.1252   
## [3] LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                          
## [5] LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## loaded via a namespace (and not attached):
## [1] digest_0.6.4     evaluate_0.5.5   formatR_0.10     htmltools_0.2.4 
## [5] knitr_1.6        rmarkdown_0.2.49 stringr_0.6.2    tools_3.1.1     
## [9] yaml_2.1.13
```

### B.2. Download and read the dataset

```r
urlTrain<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
filenameTrain = "pml-training.csv"
urlTest <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
filenameTest = "pml-testing.csv"

download.file(urlTrain, destfile=filenameTrain)
download.file(urlTest, destfile=filenameTest)
dateDownloaded <- date()
dateDownloaded
```

```
## [1] "Sat Aug 23 02:47:26 2014"
```


### B.3. Read the training and test dataset

```r
Traindata <- read.csv(filenameTrain)
Testdata <- read.csv(filenameTest)
```

### B.4 Clean the dataset
Since there are many NAs in some columns of `Traindata` and `Testdata`, we first clearning these columns.

```r
NAs <- apply(Traindata, 2, function(x) {sum(is.na(x))})
cleanTrain <- Traindata[, which(NAs == 0)]
cleanTest <- Testdata[, which(NAs == 0)]
```

### B.5. Subset the dataset
After reading the feature extraction in the paper [Qualitative Activity Recognition of Weight Lifting Exercises][4], we keep 16 variables below as predictors and apply **Random Forest Classification** algorithm to predict **personal activiy**.

```r
# names(cleanTrain)
input_vars_list <- c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", 
                     "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", 
                     "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", 
                     "total_accel_dumbbell", "roll_forearm", "pitch_forearm", 
                     "yaw_forearm", "total_accel_forearm", "classe");
cleanTrain <- cleanTrain[, input_vars_list]
cleanTest <- cleanTest[, input_vars_list[1:16]]
```

### B.6. Split the origin train dataset into training & cross-validation dataset 
To evaluate the performance of fitted model, we split 40% of origin train data as cross-validation dataset and use the rest to fit model.

```r
set.seed(1)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
inTrain <- createDataPartition(y=cleanTrain$classe,
                               p=0.7, list=FALSE)
training <- cleanTrain[inTrain,]
validation <- cleanTrain[-inTrain,]
```

### B.7. Fit Random Forest Classification Model

```r
set.seed(2)
startTime <- Sys.time();
rfControl <- trainControl(method ="cv", number=5)
modelFit <- train(classe ~.,data=training, method="rf",
                  trControl = rfControl)
endTime <- Sys.time();
endTime - startTime;
```

```
## Time difference of 3.987 mins
```

```r
modelFit
```

```
## Random Forest 
## 
## 13737 samples
##    16 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 10989, 10990, 10990, 10991, 10988 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.002        0.003   
##   9     1         1      0.003        0.003   
##   20    1         1      0.003        0.004   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 9.
```

```r
predTrain <- predict(modelFit, training)
table(predTrain,training$classe)
```

```
##          
## predTrain    A    B    C    D    E
##         A 3906    0    0    0    0
##         B    0 2658    0    0    0
##         C    0    0 2396    0    0
##         D    0    0    0 2252    0
##         E    0    0    0    0 2525
```

# C. Results

### C.1. The Error Rate of the Validation Set.

```r
predVali <- predict(modelFit, validation)
table(predVali,validation$classe)
```

```
##         
## predVali    A    B    C    D    E
##        A 1661    6    1    0    0
##        B    8 1120    3    0    3
##        C    3   10 1017    6    2
##        D    0    2    5  955    2
##        E    2    1    0    3 1075
```

```r
sampleError <- sum(predVali == validation$classe)/nrow(validation)
sampleError
```

```
## [1] 0.9903
```
Therefore, the out of sample error estimated with cross-validation is **0.9903**.

### C.2. The Prediction of the Test Set.

```r
answers <- predict(modelFit, cleanTest)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


[1]:https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
[2]:https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
[3]:http://groupware.les.inf.puc-rio.br/har
[4]:http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf

