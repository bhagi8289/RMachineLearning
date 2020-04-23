#Using Bags of Words and Naive Bayes for SMS spam classification
sms_raw = read.csv("https://raw.githubusercontent.com/bhagi8289/datasets/master/sms_spam.csv",stringsAsFactors = FALSE)
str(sms_raw)
sms_raw$type <- factor(sms_raw$type)
str(sms_raw)
table(sms_raw$type)

# Data Cleaning for Text Data
# Use text-mining package tm
# Create corpus - a collection of text documents i.e. collection of SMS messages
library(tm)
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
str(sms_corpus)
print(sms_corpus)
inspect(sms_corpus[1:2])
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:3], as.character)
# Lower the characters using transformation
sms_corpus_clean <- tm_map(sms_corpus,content_transformer(tolower))
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])
# Remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean,removeNumbers)
#> getTransformations()
# "removeNumbers"     "removePunctuation" "removeWords"       "stemDocument"      "stripWhitespace"  
# Remove stopwords
sms_corpus_clean <- tm_map(sms_corpus_clean,removeWords,stopwords())
# Remove punctuations
sms_corpus_clean <- tm_map(sms_corpus_clean,removePunctuation)
# Stemming- reduce words to their root form
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))
sms_corpus_clean <- tm_map(sms_corpus_clean,stemDocument)
# Remove white spaces
sms_corpus_clean <- tm_map(sms_corpus_clean,stripWhitespace)
as.character(sms_corpus[1:3])
as.character(sms_corpus_clean[1:3])

# Tokenization - split sentences to tokens
# Create Document Term Matrix - rows are documents and columns are terms
# We can also use Term Document Matrix which is a transpose of DTM
# Each cell of matrix is a number that represents the count of the word 
# in the column in the document represented in the row
# Create DTM sparse matrix
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm2 <- DocumentTermMatrix(sms_corpus_clean, control = list(tolower = TRUE,
                                                                removeNumbers = TRUE,
                                                                stopwords = TRUE,
                                                                removePunctuation = TRUE,
                                                                stemming = TRUE
))
sms_dtm
sms_dtm2

# Create training and test datasets
sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]
sms_train_labels <- sms_raw[1:4169,]$type
sms_test_labels <- sms_raw[4170:5559,]$type
# Compare the proportion of spam in training and test datasets
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))
# Visualize Text Data through Word Cloud
library(wordcloud)
wordcloud(sms_corpus_clean,min.freq = 50,random.order = FALSE)
# Visualize spam and non-spam word clouds
spam <- subset(sms_raw,type == "spam")
ham <- subset(sms_raw,type == "ham")
wordcloud(spam$text,max.words = 40,scale = c(3,0.5))
wordcloud(ham$text,max.words = 40,scale = c(3,0.5))
# Create indicator features for Naive Bayes
str(sms_dtm)
# Total 6542 features for words as number of columns
# Each word is a feature because it can help in detecting if an SMS is a spam or ham
# However, we eliminate words that appear in less than 5 messages or less than 0.1% of records 
# in training data
# We use findFreqTerms function to find frequent words as a character vector
findFreqTerms(sms_dtm_train,5)
sms_freq_words <- findFreqTerms(sms_dtm_train,5)
str(sms_freq_words)
# We filter the words in document term matrix that appear 5 times or more
sms_freq_dtm_train <- sms_dtm_train[,sms_freq_words]
sms_freq_dtm_test <- sms_dtm_test[,sms_freq_words]
str(sms_freq_dtm_train)
str(sms_freq_dtm_test)
# We finally have 1137 terms which will act as features for Naive Bayes
# Naive Bayes is trained on categorical variables, so we will change the terms frequency
# to "Yes" or "No" i.e. whether a term appears of not
# This approach loses the important information about the frequency of words
convert_counts <- function(x) { x <- ifelse(x>0, "Yes", "No")}
convert_counts(c(5,3,7))
# We use apply function with margin 2 (column) to apply convert_counts
sms_train <- apply(sms_freq_dtm_train,MARGIN = 2,convert_counts)
sms_test <- apply(sms_freq_dtm_test,MARGIN = 2,convert_counts)
sms_train
sms_test
str(sms_train)
sms_train[which(sms_train=="Yes")]
sms_test[which(sms_test=="Yes")]
table(sms_train)
table(sms_test)
# Use Naive Bayes implementation from e1071 package by Viena Institute of Technology
library(e1071)
# We can also use NaiveBayes from klaR package 
#Train the classifier
sms_classifier <- naiveBayes(sms_train,sms_train_labels)
# Predict on test set
sms_test_pred <- predict(sms_classifier,sms_test)
# Evaluation
library(gmodels)
CrossTable(sms_test_pred,sms_test_labels,prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('predicted','actual'))
table(sms_test_pred)
table(sms_test_labels)
# We note
# Our classifier predicted 1231 ham whereas only 1201 were actually ham, so 30 spam messages
# were incorrectly reported as ham
# Our classifier predicted 159 messages as spam wherein only 153 were actually spam and 6 ham 
# messages were incorrectly reported as spam
# Total incorrectly classified messages: 6+30 = 36
# We can improve model performance by setting Laplace smoothing parameter to 1
#Train the classifier with Laplace 1
sms_classifier <- naiveBayes(sms_train,sms_train_labels,laplace = 1)
# Predict on test set
sms_test_pred <- predict(sms_classifier,sms_test)
# Evaluation
library(gmodels)
CrossTable(sms_test_pred,sms_test_labels,prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('predicted','actual'))
table(sms_test_pred)
table(sms_test_labels)

# We notice that our model performance worsened. Therefore, Laplace smoothing did not help.