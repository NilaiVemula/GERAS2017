#loading libraries --------------------
library(tensorflow)
library(caret)
#loading functions ----------------------------
sapply(list.files(pattern="[.]R$", path="source", full.names=TRUE, recursive = F), source)

#loading data -----------------------
myData <- read.csv("~/Documents/GitHub/GERAS2017/Data/GSE109881_Zf_GERASStages_TPM.csv", header = T, row.names = 1, check.names = F)
dim(myData) # dimensions should be 31916 by 637

#extract metadata -----------------------
## Extract name from column name
myTimeString = regmatches(colnames(myData),regexpr("[0-9A-Za-z]+",colnames(myData)))
## Define the parametes for ages in sample
## mySampleAges is the age of each sample, as factor
## myTimePeriod is the period of time used: hour/day/month/year/mpf
## myTimePoints are the discrete time points present in the samples
mySampleAges = as.factor(as.numeric(regmatches(myTimeString,regexpr("[0-9]+",myTimeString))))
myTimePeriod = regmatches(myTimeString[1],regexpr("[A-Za-z]+",myTimeString[1]))
myTimePoints = levels(mySampleAges)
print(paste("The data contains",ncol(myData),"samples from",nlevels(mySampleAges),"time points, from", min(as.numeric(myTimePoints)),myTimePeriod,"to",max(as.numeric(myTimePoints)),myTimePeriod))

#classification stages ---------------------------------
## In here, we will set the stage of each discrete time point. We have seven time points: 1, 3, 4, 6, 10, 12, 14 mpf
## The first one (1 mpf) is set to 'Juvenile', next three (3, 4, 6 mpf) to 'Young', and the last three (10, 12, 14 mpf) to 'Old'
Age.labels <- c("Juvenile", rep("Young", 3), rep("Old", 3))
## We redo the old ages to new stages
## Since this command throws a warning about repeating levels, we will temporally suppress them.
suppressWarnings(Sample.age.conditions <- factor(factor(mySampleAges, labels = Age.labels)))
## And extract the number of conditions (stages). This helps with classification.
myConditions <- factor(Sample.age.conditions, labels = 0:(nlevels(Sample.age.conditions)-1))
myLevels = levels(myConditions)
cat(paste("The data contains ",nlevels(myConditions), "classification stages of: "), paste(myTimePoints, myTimePeriod, ":", Age.labels), sep = "\n")

#Building the classifier ---------------
## Now, we will choose the genes that will be used to build the classifier. We will do so by choosing the top most variable genes in the test dataset. Gene selection can also be done based to genes with highest variation, median absolute deviation (MAD), mean, median.
## Choose the variability parameter: "var", "mad", "mean", "median"
myVariance <- "mad"
## Choose the number of parameters (genes) for the model
myTopGenes <- 1000
if (myVariance == "var") {
  vars = apply(log(myData+1),1,var)
} else if (myVariance == "mean") {
  vars = apply(log(myData+1),1,mean)
} else if (myVariance == "median") {
  vars = apply(log(myData+1),1,median)
} else {
  vars = apply(log(myData+1),1,mad)
}
idx = order(vars, decreasing = TRUE)
feat.Selected = idx[1:myTopGenes]
myData.Selected = myData[feat.Selected,]
parameters.name = rownames(myData.Selected)
## A few of the parameters (genes) whose expression values will be used by the model
head(parameters.name)

#partitioning data ----------------
## We will partition the data into train and test set for building the classifier
## 80% of data will be used for training. The rest 20% will be kept aside for testing the performance of the model
ratio=0.8
set.seed(171)
n <- ncol(myData.Selected)
ind = unlist(sapply(split(1:length(myConditions), myConditions),
                    function(x) sample(x, size = ratio*length(x), replace = F)),
             use.names = F)
train.Selected = myData.Selected[,ind]+1
test.Selected = myData.Selected[,-ind]+1
train.cond = myConditions[ind]
test.cond = myConditions[-ind]
train.log = apply(train.Selected, 2, function(x) log(x, 2))
test.log = apply(test.Selected, 2, function(x) log(x, 2))
## To know how the training and test datalooks
table(train.cond)
table(test.cond)

#learning rate -------------
## Too high learning rate
parameters.weights.learn = model_dropout(X_train = train.log, Y_train = train.cond,
                                         num_epochs = 50, learning_rate = 1,
                                         print_cost = T)
## Too low learning rate
parameters.weights.learn = model_dropout(X_train = train.log, Y_train = train.cond,
                                         num_epochs = 50, learning_rate = 1e-08,
                                         print_cost = T)
## Just right learning rate
parameters.weights.learn = model_dropout(X_train = train.log, Y_train = train.cond,
                                         num_epochs = 50, learning_rate = 0.001,
                                         print_cost = T)

#cross-validation -----------------
