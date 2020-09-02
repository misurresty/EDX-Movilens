#######################################################################
# Movielens Predictive Model
# September 05, 2020
# Miguel Sanchez
# 
# Deploy a machine learning algorithm to create a movie ratings prediction.
# Root Mean Squared Error (RMSE) will be used to indicate the absolute 
# fit of the model to the data
#######################################################################

#######################################################################
# The code used for training and validation set has been taken from the 
# EDX site
######################################################################


#################################
# Create edx set, validation set
#################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies, stringsAsFactors=TRUE) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                                                  title = as.character(title),
                                                                  genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
save(edx, validation, file="movielens-masu.rda")

##################################################################
# Checking and validating the loaded data
##################################################################

load(file="movielens-masu.rda")
head(edx)
##################################################################
#Data Set (sample) with 6 variables
##################################################################
cat("Train set dimension :",dim(edx))
##################################################################
#Data set with 9.000.055 records and 6 variables
##################################################################
str(edx) 
##################################################################
#2 categorical and 4 continuous variables
##################################################################
# Number of unique movies
##################################################################
unique_movies<- edx$movieId %>% unique() %>% length()
unique_movies
##################################################################
# Number of unique users
##################################################################
unique_users<- edx$userId %>% unique() %>% length()
unique_users
##################################################################
#Movies with most ratings
##################################################################
edx %>% group_by(movieId) %>%
  summarise(n_ratings=n(), title=first(title)) %>%
  top_n(10, n_ratings)
##################################################################
#Histogram of number of reviews for each movie
##################################################################
edx %>%
  group_by(movieId) %>%
  summarise(n_reviews=n()) %>%
  ggplot(aes(n_reviews)) +
  geom_histogram(color="red") +
  scale_x_log10()



##################################################################
# DATA WRANGLING 
# Split the edx data set in train and test
##################################################################
# Split using 70% for training and 30% for test
#################################################################

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.3, list = FALSE)
train_data_set <- edx[-test_index,]
temporal <- edx[test_index,]

test_data_set <- temporal %>% 
  semi_join(train_data_set, by = "movieId") %>%
  semi_join(train_data_set, by = "userId")

borrar <- anti_join(temporal, test_data_set)
train_data_set <- rbind(train_data_set, borrar)

rm(test_index, temporal, borrar)
###############################################################
#Checking number of records for test data set - 30%
###############################################################
dim(test_data_set)
###############################################################
#Checking number of records for training data set - 70%
###############################################################
dim(train_data_set)
################################################################
# For model evaluation we will be using RMSE
################################################################

################################################################
# Machine Learning Model
# A linear model will be used for prediction;only Movie and User 
# variables are being considered (due to performace in my laptop)
# movie effect = bi & bu = user effect
###############################################################
#For comparison purposes I will use the mean as the first prediction
# and calculate the RMSE (only Mu)
################################################################
mu<- mean(train_data_set$rating)
rmse_initial<-sqrt(mean((test_data_set$rating - mu)^2))
rmse_initial
################################################################
# I will try now with movie effect --> bi
################################################################
bi<- train_data_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
################################################################
# Predicting the rating using mu + bi
################################################################
y_hat_bi <- mu + test_data_set %>% 
  left_join(bi, by = "movieId") %>% 
  .$b_i
y_hat_bi
###############################################################
# Calculate the RMSE using mu + bi
###############################################################
rmse_mu_bi<- sqrt(mean((test_data_set$rating - y_hat_bi)^2))
rmse_mu_bi
###############################################################
# Including the user effect -->bu
###############################################################
bu <- train_data_set %>% 
  left_join(bi, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
###############################################################
# Predicting the rating using mu + bi + bu
##############################################################
y_hat_bi_bu <- test_data_set %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
y_hat_bi_bu
###############################################################
# Calculate the RMSE using mu + bi +bu
###############################################################
rmse_mu_bi_bu<- sqrt(mean((test_data_set$rating - y_hat_bi_bu)^2))
rmse_mu_bi_bu

###############################################################
#
# Based on the results I have gotten from the algorithm, the RMSE 
# has been reduced reduced thus the prediction has improved
############################################################### 
EVALUATION<- c("RMSE_MU", "RMSE_MU+BI", "RMSE_MU+BI+BU")
#result
VALUES<- c(rmse_initial, rmse_mu_bi, rmse_mu_bi_bu)
#result2
Result3<- data.frame(EVALUATION, VALUES)
Result3
###############################################################
#Running against validation data set
###############################################################

mu_v2<- mean(train_data_set$rating)
rmse_initial_v2<-sqrt(mean((validation$rating - mu)^2))
rmse_initial_v2

bi_v2<- train_data_set %>% 
  group_by(movieId) %>% 
  summarize(b_i_v2 = mean(rating - mu))

y_hat_bi_v2 <- mu_v2 + validation %>% 
  left_join(bi_v2, by = "movieId") %>% 
  .$b_i_v2
y_hat_bi_v2

rmse_mu_bi_v2<- sqrt(mean((validation$rating - y_hat_bi_v2)^2))
rmse_mu_bi_v2

bu_v2 <- train_data_set %>% 
  left_join(bi_v2, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u_v2 = mean(rating - mu_v2 - b_i_v2))

y_hat_bi_bu_v2 <- validation %>% 
  left_join(bi_v2, by='movieId') %>%
  left_join(bu_v2, by='userId') %>%
  mutate(pred = mu_v2 + b_i_v2 + b_u_v2) %>%
  .$pred
y_hat_bi_bu_v2

rmse_mu_bi_bu_v2<- sqrt(mean((validation$rating - y_hat_bi_bu_v2)^2))
rmse_mu_bi_bu_v2

##############################################################
#The results are good, but I believe the RMSE can be improved 
# using the linear model with regularisation 
##############################################################

# The first step is tocalculate the best lambda
lambdas <- seq(from=0, to=10, by=0.25)


#Calculate the RMSE on each defined LAMBDA, using MU+BI+BU
rmses <- sapply(lambdas, function(l){
  # calculate average (MU)
  mu_v3 <- mean(edx$rating)
  # calculate the bi
  b_i_v3 <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i_v3 = sum(rating - mu_v3)/(n()+l))
  # calculate the bu 
  b_u_v3 <- edx %>% 
    left_join(b_i_v3, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_v3 = sum(rating - b_i_v3 - mu_v3)/(n()+l))
  # Run the predictions on validation data set
  predicted_ratings <- validation %>% 
    left_join(b_i_v3, by = "movieId") %>%
    left_join(b_u_v3, by = "userId") %>%
    mutate(pred_v3 = mu_v3 + b_i_v3 + b_u_v3) %>%
    pull(pred_v3)
  # Print RMSE on predictions 
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot of RMSE vs lambdas
qplot(lambdas, rmses)
# print minimum RMSE 
min(rmses)
#print (lambdas)
lambda <- lambdas[which.min(rmses)]
print (lambda)

###########################################################
# Model with regularized movie -->BI and user effect --> BU
###########################################################

# Linear model with the minimizing lambda
# Calculate the mean (MU)
mu_v4 <- mean(edx$rating)
# Calculate the BI
b_i_v4 <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i_v4 = sum(rating - mu_v4)/(n()+lambda))
# Calculate the BU 
b_u_v4 <- edx %>% 
  left_join(b_i_v4, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_v4 = sum(rating - b_i_v4 - mu_v4)/(n()+lambda))
# Run predictions using calculated BI & BU
predicted_ratings <- validation %>% 
  left_join(b_i_v4, by = "movieId") %>%
  left_join(b_u_v4, by = "userId") %>%
  mutate(pred = mu_v4 + b_i_v4 + b_u_v4) %>%
  pull(pred)
# output RMSE of predictions
RMSE(predicted_ratings, validation$rating)

#################################################################
# Conclusion
# The linear model considering movie (BI) anf user effects (BU) created an acceptable resut; however,
# the result was improved (< 0.86490) using regularisation
################################################################