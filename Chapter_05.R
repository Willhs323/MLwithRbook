##### Chapter 5: Classification using Decision Trees and Rules -------------------
setwd("/Users/wills/Documents - PC/R resources/ML with R/")
#### Part 1: Decision Trees -------------------

## Understanding Decision Trees ----
# calculate entropy of a two-class segment
-0.60 * log2(0.60) - 0.40 * log2(0.40)

curve(-x * log2(x) - (1 - x) * log2(1 - x),
      col = "red", xlab = "x", ylab = "Entropy", lwd = 4)

## Example: Identifying Risky Bank Loans ----
## Step 2: Exploring and preparing the data ----
credit <- read.csv("credit.csv", stringsAsFactors = TRUE) # TRUE becasue its all categorical
str(credit)

# look at two characteristics of the applicant
table(credit$checking_balance)
table(credit$savings_balance)

# look at two characteristics of the loan
summary(credit$months_loan_duration)
summary(credit$amount) # DM is Deutsche mark

# look at the class variable
table(credit$default)

# create a random sample for training and test data
# use set.seed to use the same random number sequence as the tutorial
set.seed(9829) # it is not random order so use sample and set seed
train_sample <- sample(1000, 900) # use 90% as training data not 75 as smaller size

str(train_sample) # vector of 900 inteagers

# split the data frames - use your random numbers you made to sample
credit_train <- credit[train_sample, ]
credit_test  <- credit[-train_sample, ]

# check the proportion of class variable - you have about 30% of those that defaulted in your dataset
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

## Step 3: Training a model on the data ----
# build the simplest decision tree - c5.0 is a algorithm
library(C50)
# m <- c5.0(class ~ predictors, data = mydata, trials = 1, costs = NULL)
# class - column from dataset to be predicted
# predictors, can use multiple with +. y ~ . makes it use all of the predictors
# Trials - optional, control number of boosting iterations
# costs - optional matrix, specifying costs with error types
credit_model <- C5.0(default ~ ., data = credit_train) # predict if it defaults using all of them

# display simple facts about the tree
credit_model # 57 decisions deep

# display detailed information about the tree
summary(credit_model)

# if balance >200 not likely to default, then if 0-200, checks loan duration, employment duration, etc
# first line 405/53 shows - of 415 examples reaching that decision, 55 incorrectly classified as not likely to default
# then a confusion matrix occurs
# it will overfit. the error here is likely better than you'd expect on the test data
# attribute usage - shows the most important predictors - the pct of rows using this to make a prediction
x11()
plot(credit_model) # use big screen. very big

## Step 4: Evaluating model performance ----
# create a factor vector of predictions on test data
# p <- predict(m, test, type = "class")
# m is model by c5.0()
# test is dataframe with test data and same features
# type is class or probability
# returns a vector of predicted class values or raw probabilities
credit_pred <- predict(credit_model, credit_test) # makes a vector of predicted variables which we will look at w/ crosstables


# cross tabulation of predicted versus actual classes
library(gmodels)
# prop.c and prop.r = FALSE - removes column and row percentages
# prop.t - the remaining one - shows the overall percent from 100 here
CrossTable(credit_test$default, credit_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# we show it correctly predicted 79%. it only predicted 9 defaults correctly
# if it predicted no default 100% of the time it would be 70% accurate

## Step 5: Improving model performance ----

## Boosting the accuracy of decision trees
# adaptive boosting - make many decision trees, and trees vote on best example in each case
# boosted decision tree with 10 trials
# combine many weak ones to make a good team
# trials sets an upper limit - number of separate decision trees to use. it will stop if nth+1 doesn't help
# start w/ 10 trials - teh standard as research suggests this reduces error rates on test daata by about 25%
credit_boost10 <- C5.0(default ~ ., data = credit_train,
                       trials = 10)
credit_boost10 # avg tree size is not 48.6. it boosted 10x
summary(credit_boost10) # shows all 10. there are subtrees denoted s#
# now it makes 27 mistakes on the 900 training data

credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
# we reduced from 21% error rate to 20
# boosting isn't as helpful if its noisy
# model still only predicts defaults on 12/25 actual ones

## Making some mistakes more costly than others
# we want to reduce false negatives so bank won't lose money
# we will make a cost matrix to make a penalty - how more costly each error is to the other

# create dimensions for a cost matrix
matrix_dimensions <- list(c("no", "yes"), c("no", "yes"))
names(matrix_dimensions) <- c("predicted", "actual")
matrix_dimensions

# build the matrix
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2, dimnames = matrix_dimensions) # r fills by columns
error_cost # say a loan default costs the bank 4x as much as a missed opportunity
# no cost w/ a correct answer

# apply the cost matrix to the tree
credit_cost <- C5.0(default ~ ., data = credit_train,
                          costs = error_cost)
credit_cost_pred <- predict(credit_cost, credit_test)

CrossTable(credit_test$default, credit_cost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
# more mistakes overall - 42% error rate however most actual defaults are correctly predicted now


#### Part 2: Rule Learners -------------------
# rule learners - closely related to decision tree learners
# if-else structure
# don't have to follow the whole tree
# use when features are entirely nominal. they do well at finding rare events
# separate and conquor. divide into smaller groups until everything is grouped
# you can use a rule at the end for the last data for an example

# the 1R algorithm
# zeroR - no rules, just predicts the most common class
# 1R selects a single rule. does surprisingly well but simplistic
# then in 1R, you select on the majority class after the first division

# RIPPER algorithm
# IREP - incremental reduced error pruning algorithm - prepruning and postpruning rules, complex
# RIPPER - repeated incremental pruning to produce error reduction
# similar to decision trees, not great for numeric data though
# grow-prune-optimize. perfectly classifies it via info gain, then prunes rule when increasing its specificity no longer reduces entropy
# can make rules with more than one feature

# if you specify rules=TRUE in c5.0() it will make a model using classification rules
# trees are divide and conquor. rules are separate and conquor
# d&c - once a rule is made you cant go back
# s&c is more dynamic - you can re-conquer old rules made in rule learners
# they are both greedy learners - they keep going until everything is classified. downside is it might not be most optimal

## Example: Identifying Poisonous Mushrooms ----
## Step 2: Exploring and preparing the data ---- 
mushrooms <- read.csv("mushrooms.csv", stringsAsFactors = TRUE)
# 8124 mushrooms
# definitely edible, definitely poisonous, likely poisonous. not recommended to be eaten was put into definitely poisonous here

# examine the structure of the data frame
str(mushrooms)
View(mushrooms)
# drop the veil_type feature - only 1 level and might have been badly extracted
mushrooms$veil_type <- NULL

# examine the class distribution
table(mushrooms$type)

# note - no predictions here. we are saying these are all kinds of poisonous mushrooms
# we are just trying to classify these not predict
# zeroR would say that theyre all edible as 52% are edible

## Step 3: Training a model on the data ----
library(OneR)

# train OneR() on the data
mushroom_1R <- OneR(type ~ ., data = mushrooms) # use all variables
# 1R finds the single best predictor
# m <- OneR(class ~ predictors, data = mydata)
# p <- predict(m, test) where m is the model, test is the test data with same features as training data

## Step 4: Evaluating model performance ----
mushroom_1R # odor was selected for rule generation. if it smells bad it is likely poisonous
# note accuracy is 98.5%
# we need 100% - any death is bad!
# make a confusion matrix of predicted vs actual values
mushroom_1R_pred <- predict(mushroom_1R, mushrooms)
table(actual = mushrooms$type, predicted = mushroom_1R_pred)
# we predicted 120 as edible that were actully poisonous
# Jrip() is a java-based implementation of RIPPER. need to download Java

## Step 5: Improving model performance ----
library(RWeka)
# m <- JRip (class ~ predictors, data = mydata)- returns a RIPPER model object
# p <- predict(m, test) where m is model data, test is data with same features
mushroom_JRip <- JRip(type ~ ., data = mushrooms)
mushroom_JRip # 9 total rules. think as if-else statements
# if odor is foul its poisonous. if gill zie is narrow, color is buff - poisonous, etc
# lastly - else its edible
# numbers are correct/incorrect - there are no incorrect ones here

summary(mushroom_JRip)

# Rule Learner Using C5.0 Decision Trees (not in text)
library(C50)
mushroom_c5rules <- C5.0(type ~ ., data = mushrooms, rules = TRUE)
summary(mushroom_c5rules)


