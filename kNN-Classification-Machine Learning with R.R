#k-NN classification in R
wbcd <- read.table("https://raw.githubusercontent.com/bhagi8289/datasets/master/wisc_bc_data.csv",sep = ',',header=TRUE, stringsAsFactors = TRUE)
wbcd

str(wbcd)
wbcd <- wbcd[-1]
str(wbcd)
table(wbcd$diagnosis)
wbcd$diagnosis <- factor(wbcd$diagnosis,levels=c("B","M"),labels = c("Benign","Malignant"))
table(wbcd$diagnosis)
round(prop.table(table(wbcd$diagnosis))*100,digits = 1)
summary(wbcd[c("radius_mean","area_mean","smoothness_mean")])
normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}
normalize(c(1,2,3,4,5))
normalize(c(10,20,30,40,50))
wbcd_n <- as.data.frame(lapply(wbcd[2:31],normalize))
str(wbcd_n)
summary(wbcd_n[c("radius_mean","area_mean","smoothness_mean")])
wbcd_train <- wbcd_n[1:469,]
wbcd_test <- wbcd_n[470:569,]
wbcd_train_labels <- wbcd[1:469,1]
wbcd_test_labels <- wbcd[470:569,1]
library(class)
wbcd_pred <- knn(train = wbcd_train,test = wbcd_test, cl = wbcd_train_labels, k=21)
wbcd_pred
library(gmodels)
CrossTable(x=wbcd_test_labels,y=wbcd_pred,prop.chisq = FALSE)
table(wbcd_test_labels)
table(wbcd_pred)
# To improve model performance, instead of min-max scaling, we use z-score standardization
# we use built-in scale() function
wbcd_z <- as.data.frame(scale(wbcd[-1]))
summary(wbcd_z[c("radius_mean","area_mean","smoothness_mean")])
wbcd_train <- wbcd_z[1:469,]
wbcd_test <- wbcd_z[470:569,]
wbcd_train_labels <- wbcd[1:469,1]
wbcd_test_labels <- wbcd[470:569,1]
library(class)
wbcd_pred <- knn(train = wbcd_train,test = wbcd_test, cl = wbcd_train_labels, k=21)
wbcd_pred
library(gmodels)
CrossTable(x=wbcd_test_labels,y=wbcd_pred,prop.chisq = FALSE)
table(wbcd_test_labels)
table(wbcd_pred)
# The result has a worse performance
