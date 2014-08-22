
# The encoding of the script is CP916
# R version 3.1.1 (2014-07-10) -- "Sock it to Me"
# Platform: i386-w64-mingw32/i386 (32-bit)
# Operating system: Windows 7 (SP1)
# 20140822 qingcheng

# Couresra-[Practical Machine Learning][predmachlearn-004][Course Project 1]

# In this project, we will analysis data from accelerometers on the belt, forearm, 
# arm, and dumbell of 6 participants. They were asked to perform barbell lifts 
# correctly and incorrectly in 5 different ways. Then build a machine learning 
# model to predict the manner in which they did the exercise. 


# change the locale in English environment
Sys.setlocale('LC_ALL', 'English')
sessionInfo()

# Download the dataset
urlTrain<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
filenameTrain = "pml-training.csv"
urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
filenameTest = "pml-testing.csv"
urlInfor <- "http://groupware.les.inf.puc-rio.br/har"

if (file.exists(filenameTrain)){
    print("Training data file already exists locally")
}else{
    print("Training data file not found locally, so download from website")
    download.file(urlTrain, destfile=filenameTrain)
    download.file(urlTest, destfile=filenameTest)
    dateDownloaded <- date()
    dateDownloaded}
#     [1] "Fri Aug 22 23:12:23 2014"
}

# Read the training and test dataset
Traindata <- read.csv(filenameTrain)
Testdata <- read.csv(filenameTest)

# Clearning dataset
NAs <- apply(Traindata, 2, function(x) {sum(is.na(x))})
cleanTrain <- Traindata[, which(NAs == 0)]
cleanTest <- Testdata[, which(NAs == 0)]

# Subset the data set
# names(cleanTrain)
input_vars_list <- c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", 
                     "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", 
                     "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", 
                     "total_accel_dumbbell", "roll_forearm", "pitch_forearm", 
                     "yaw_forearm", "total_accel_forearm", "classe");
cleanTrain <- cleanTrain[, input_vars_list]
cleanTest <- cleanTest[, input_vars_list[1:16]]


# Split the clearning train data into training & cross-validation dataset 
set.seed(1)
library(caret)
inTrain <- createDataPartition(y=cleanTrain$classe,
                               p=0.7, list=FALSE)
training <- cleanTrain[inTrain,]
validation <- cleanTrain[-inTrain,]

# print(object.size(training), units = "MB")

# Fit model
set.seed(2)
startTime <- Sys.time();
rfControl <- trainControl(method ="cv", number=5)
modelFit <- train(classe ~.,data=training, method="rf",
                  trControl = rfControl)
endTime <- Sys.time();
endTime - startTime;

modelFit

predTrain <- predict(modelFit, training)
table(predTrain,training$classe)


# Calculation the errors using the Validation Set.
predVali <- predict(modelFit, validation)
table(predVali,validation$classe)
sampleError <- sum(predVali == validation$classe)/nrow(validation)
sampleError

answers <- predict(modelFit, cleanTest)

# Submission
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)

