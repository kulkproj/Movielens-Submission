
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

#Start timer
start_time <- Sys.time()
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################
#10/01/2022
# Note: this process may take a 15-25 minutes
start_time <- Sys.time()
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

#set up variable vector for version number
v<-as.numeric(substr(R.Version()$version.string,11,13))
#populate movies according to version
if (v >= 4.0) {
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                             title = as.character(title),
                                             genres = as.character(genres))} else {  
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                            title = as.character(title),
                                            genres = as.character(genres))}

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
if (v <= 3.5) {
  set.seed(1)} else {  
    set.seed(1, sample.kind="Rounding")}
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
#while debugging display elapsed time, list column names of training set
#Sys.time() - start_time
#colnames(edx)
colnames(validation)
#columns in data frame
# userId                    feature
# movieId                   feature
# rating                    outcome
# timestamp                 feature ?
# title
# genres (pipe delimited)   feature

#Exploring various methods RMSE lower than 1 
# and as low as possible is being sought

#lm() can't be used because very large dataset

#naive approach

mu <- mean(edx$rating)
#mu #3.512465

rmse_naive <- RMSE(validation$rating, mu)
#rmse_naive # naive 1.061205

#Movie Effects
mu <- mean(edx$rating)

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

#plot almost binomial normal

predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
rmse_movie<-RMSE(predicted_ratings, validation$rating)  # Movie effect 0.9439049
#rmse_movie

#User Effect

edx%>%
  group_by(userId) %>%
  summarize(b_u = mean(rating)) %>%
  filter(n()>=100) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black")

#plot almost binomial normal


user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
rmse_movie_and_user<-RMSE(predicted_ratings, validation$rating) # movie & user 0.8653458
#rmse_movie_and_user 

#Regularisation to penalise noisy movie effect data - 
#movies with low number of ratings

lambda <- 3 #formula p648 of course text
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

tibble(original = movie_avgs$b_i,
       regularlized = movie_reg_avgs$b_i,
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) +
  geom_point(shape=1, alpha=0.5)

predicted_ratings <- validation %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
rmse_regularised<-RMSE(predicted_ratings, validation$rating) #0.9438546
#rmse_regularised

# tune lambda to find best value

lambdas <- seq(0, 10, 0.25)
mu <- mean(edx$rating)
rating_sum <- edx %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>%
    left_join(rating_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})
qplot(lambdas, rmses)

lambda<-lambdas[which.min(rmses)] #minimum lambda movie = 2.25

rating_sum <- edx %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())

rmse_tuned_movie <- sapply(lambda, function(l){
  predicted_ratings <- validation %>%
    left_join(rating_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})
#rmse_movie_min_lambda #0.9438523


lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <-
    validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})
qplot(lambdas, rmses)
lambda<-lambdas[which.min(rmses)] #minimum lambda movie+user  5.25

rmse_tuned_movie_user <- sapply(lambda, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <-
    validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

#attempt genre
genre_train <- edx %>% separate_rows(genres, sep = "\\|") 
genre_test <- validation %>% separate_rows(genres, sep = "\\|") 
genre_edx <- edx %>% separate_rows(genres, sep = "\\|") 
genre_validation <- validation %>% separate_rows(genres, sep = "\\|") 



#lambdas <- seq(0, 10, 0.25) #min not found <10.0 
lambdas <- seq(8, 18, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(genre_train$rating)
  b_i <- genre_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- genre_train %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- genre_train %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  predicted_ratings <-
    genre_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  return(RMSE(predicted_ratings, genre_test$rating))
})
qplot(lambdas, rmses)
lambda_umg<-lambdas[which.min(rmses)] #minimum lambda movie+user  5.25

rmse_tuned_umg <- sapply(lambda_umg, function(l){
  mu <- mean(genre_train$rating)
  b_i <- genre_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- genre_train %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- genre_train %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  predicted_ratings <-
    genre_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  return(RMSE(predicted_ratings, genre_test$rating))
})
#rmse_tuned_umg


rmse_naive 
rmse_movie
rmse_movie_and_user 
rmse_regularised
rmse_tuned_movie
rmse_tuned_movie_user
rmse_tuned_umg
#timestamp->date->recent rating measure as a predictor?

Sys.time() - start_time
