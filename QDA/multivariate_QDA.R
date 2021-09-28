# Quadratic discriminant analysis in a multivariate setting with three classes

# Load packages and read data
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, MASS)

#####
#. Step 1: Data generation

# Generate sample data for three outcomes (0, 1, and 2)

# We will use two predictors in this example, using MASS's `mvrnorm` function
# to sample from a multivariate normal distribution. `mu` will be our vector of
# population means for the predictors. Our predictors will be correlated with
# one another, which will be reflected in the covariance matrix `sigma`

# Choose population means per group, centered on (X1, X2, X3) coordinate means

pop_mean_c0_X1 <- sample.int(10, 1)
pop_mean_c1_X1 <- sample.int(10, 1)
pop_mean_c2_X1 <- sample.int(10, 1)

pop_mean_c0_X2 <- sample.int(10, 1)
pop_mean_c1_X2 <- sample.int(10, 1)
pop_mean_c2_X2 <- sample.int(10, 1)

mu_c0 <- c(pop_mean_c0_X1, pop_mean_c0_X2)
mu_c1 <- c(pop_mean_c1_X1, pop_mean_c1_X2)
mu_c2 <- c(pop_mean_c2_X1, pop_mean_c2_X2)

# Generate sample data: predictor variance will be on the diagonals and 
# covariance on the immediate off-diagonals. Correlation is between variables
# (X1 and X2) and variance is within-group and identical for both groups

# Note that QDA *does not* assume each class has the same covariance matrix, so
# we will create separate population covariance matrices for each pair of
# variables

pop_c0_corr <- runif(1, 0, 1)
pop_c1_corr <- runif(1, 0, 1)
pop_c2_corr <- runif(1, 0, 1)

pop_var_c0 <- runif(1, 0, 10)
pop_var_c1 <- runif(1, 0, 10)
pop_var_c2 <- runif(1, 0, 10)

# Generate sample data from multivariate normal distributions, choosing n samples

n <- 300

for (i in 0:2) {
  
  class <- paste("c", i, sep = "")
  
  class_var <- paste("pop_var", class, sep = "_")
  class_corr <- paste("pop", class, "corr", sep = "_")
  mu = paste("mu", class, sep = "_")
  
  temp_sigma <- matrix(c(get(class_var), get(class_corr),
                        get(class_corr), get(class_var)), 2, 2)
  
  temp_train_data <- mvrnorm(n = n,
                             mu = get(mu),
                             Sigma = temp_sigma)
  
  temp_test_data <- mvrnorm(n = n,
                            mu = get(mu),
                            Sigma = temp_sigma)
  
  assign(paste(class, "sigma", sep = "_"), temp_sigma)
  assign(paste(class, "train", sep = "_"), temp_train_data)
  assign(paste(class, "test", sep = "_"), temp_test_data)
  
}

train_sample_df <- bind_rows(
  tibble(y = 0,
         X1 = c0_train[,1],
         X2 = c0_train[,2]),
  tibble(y = 1,
         X1 = c1_train[,1],
         X2 = c1_train[,2]),
  tibble(y = 2,
         X1 = c2_train[,1],
         X2 = c2_train[,2])
)

test_sample_df <- bind_rows(
  tibble(y = 0,
         X1 = c0_test[,1],
         X2 = c0_test[,2]),
  tibble(y = 1,
         X1 = c1_test[,1],
         X2 = c1_test[,2]),
  tibble(y = 2,
         X1 = c2_test[,1],
         X2 = c2_test[,2])
)

# Plot 95% confidence ellipses for the groups

train_conf_ellipse <- ggplot(train_sample_df) + 
  stat_ellipse(aes(x = X1, y = X2, group = as.factor(y), color = as.factor(y))) +
  scale_color_manual(values = c("red", "blue", "green"),
                     name = "Class") +
  labs(title = "Bivariate Training Sample of Three Classes: 95% Probability Ellipses")

train_conf_ellipse

test_conf_ellipse <- ggplot(test_sample_df) + 
  stat_ellipse(aes(x = X1, y = X2, group = as.factor(y), color = as.factor(y))) +
  scale_color_manual(values = c("red", "blue", "green"),
                     name = "Class") +
  labs(title = "Bivariate Test Sample of Three Classes: 95% Probability Ellipses")

test_conf_ellipse

# Plot scatterplot of data

train_sample_scatter <- ggplot(train_sample_df) + 
  geom_point(aes(x = X1, y = X2, color = as.factor(y))) +
  scale_color_manual(values = c("red", "blue", "green"),
                     name = "Class") +
  labs(title = "Bivariate Training Sample of Three Classes: Scatterplot")

train_sample_scatter

test_sample_scatter <- ggplot(test_sample_df) + 
  geom_point(aes(x = X1, y = X2, color = as.factor(y))) +
  scale_color_manual(values = c("red", "blue", "green"),
                     name = "Class") +
  labs(title = "Bivariate Test Sample of Three Classes: Scatterplot")

test_sample_scatter

#####
# Step 2: Calculate optimal Bayes decision boundary (for comparison)

# As in the univariate case, we will build the Bayes classifier for comparison,
# but in this case the classifier is more complex and more closely resembles the
# construction of the univariate LDA classifier (after all, LDA is essentially
# an approximation of the Bayes classifier).

# We will calculate the class-score for each observation (note that this is
# likely not a stable or efficient way to do this in a finite-precision
# computational setting)

pop_c0_mean_vec <- c(pop_mean_c0_X1, pop_mean_c0_X2)
pop_c1_mean_vec <- c(pop_mean_c1_X1, pop_mean_c1_X2)
pop_c2_mean_vec <- c(pop_mean_c2_X1, pop_mean_c2_X2)

c0_prior <- n / (n * 3)
c1_prior <- n / (n * 3)
c2_prior <- n / (n * 3)

d0_bayes <- function(x_vec){
  -.5 * log(norm(c0_sigma, type = "2")) - 
    .5 * t(x_vec - mu_c0) %*% solve(c0_sigma) %*% (x_vec - mu_c0) + 
    log(c0_prior)
}

d1_bayes <- function(x_vec){
  -.5 * log(norm(c1_sigma, type = "2")) - 
    .5 * t(x_vec - mu_c1) %*% solve(c1_sigma) %*% (x_vec - mu_c1) + 
    log(c1_prior)
}

d2_bayes <- function(x_vec){
  -.5 * log(norm(c2_sigma, type = "2")) - 
    .5 * t(x_vec - mu_c2) %*% solve(c2_sigma) %*% (x_vec - mu_c2) + 
    log(c2_prior)
}

bayes_classifier <- function(x_vec){
  score_c0 <- d0_bayes(x_vec)
  score_c1 <- d1_bayes(x_vec)
  score_c2 <- d2_bayes(x_vec)
  
  if (score_c0 > score_c1 & score_c0 > score_c2) {
    0
  } else if (score_c1 > score_c0 & score_c1 > score_c2) {
    1
  } else {
    2
  }
}

# Apply the Bayes classifier to the training data

train_sample_df$bayes_predicted_y <- apply(train_sample_df[, c("X1", "X2")], 
                                           1, bayes_classifier)

#####
# Step 3: Create the LDA Classifier and Apply To Training Set

# The LDA classifier works the same way as the Bayes classifier, except we use
# the sample means and covariance matrix and not the population means and
# covariance matrix.

sample_c0_mean_vec <- c(mean(train_sample_df$X1[train_sample_df$y == 0]), 
                        mean(train_sample_df$X2[train_sample_df$y == 0]))
sample_c1_mean_vec <- c(mean(train_sample_df$X1[train_sample_df$y == 1]), 
                        mean(train_sample_df$X2[train_sample_df$y == 1]))
sample_c2_mean_vec <- c(mean(train_sample_df$X1[train_sample_df$y == 2]), 
                        mean(train_sample_df$X2[train_sample_df$y == 2]))

# In reality, the sampled prior probabilities may not be the same as the
# population prior probabilities. Here they are, so we do not technically need
# to calculate the prior probabilities again but we will for completeness.

c0_prior <- nrow(train_sample_df[train_sample_df$y == 0,]) /
  nrow(train_sample_df)
c1_prior <- nrow(train_sample_df[train_sample_df$y == 1,]) /
  nrow(train_sample_df)
c2_prior <- nrow(train_sample_df[train_sample_df$y == 2,]) /
  nrow(train_sample_df)

# Create and apply the LDA classifier, then plot the LDA decision boundaries

sample_cov_mat_c0 <- cov(train_sample_df %>% 
                           filter(y == 0) %>% 
                           dplyr::select(X1, X2))

sample_cov_mat_c1 <- cov(train_sample_df %>% 
                           filter(y == 1) %>% 
                           dplyr::select(X1, X2))

sample_cov_mat_c2 <- cov(train_sample_df %>% 
                           filter(y == 2) %>% 
                           dplyr::select(X1, X2))

# Unlike with LDA, we do not need to build a class-weighted covariance matrix
# since we are not assuming that the class-level covariance matrices are equal

# Create the LDA classifier

d0_LDA <- function(x_vec){
  -.5 * log(norm(sample_cov_mat_c0, type = "2")) - 
    .5 * t(x_vec - sample_c0_mean_vec) %*% solve(sample_cov_mat_c0) %*% 
    (x_vec - sample_c0_mean_vec) + 
    log(c0_prior)
}

d1_LDA <- function(x_vec){
  -.5 * log(norm(sample_cov_mat_c1, type = "2")) - 
    .5 * t(x_vec - sample_c1_mean_vec) %*% solve(sample_cov_mat_c1) %*% 
    (x_vec - sample_c1_mean_vec) + 
    log(c1_prior)
}

d2_LDA <- function(x_vec){
  -.5 * log(norm(sample_cov_mat_c2, type = "2")) - 
    .5 * t(x_vec - sample_c2_mean_vec) %*% solve(sample_cov_mat_c2) %*% 
    (x_vec - sample_c2_mean_vec) + 
    log(c2_prior)
}

LDA_classifier <- function(x_vec){
  score_c0 <- d0_LDA(x_vec)
  score_c1 <- d1_LDA(x_vec)
  score_c2 <- d2_LDA(x_vec)
  
  if (score_c0 > score_c1 & score_c0 > score_c2) {
    0
  } else if (score_c1 > score_c0 & score_c1 > score_c2) {
    1
  } else {
    2
  }
}

# Apply the Bayes classifier to the training data

train_sample_df$LDA_predicted_y <- apply(train_sample_df[, c("X1", "X2")], 
                                         1, LDA_classifier)

#####
# Step 4: Analyze performance

# Apply our trained Bayes and LSA models to the test data

test_sample_df$bayes_predicted_y <- apply(test_sample_df[, c("X1", "X2")], 
                                          1, bayes_classifier)

test_sample_df$LDA_predicted_y <- apply(test_sample_df[, c("X1", "X2")], 
                                        1, LDA_classifier)

# Find the misclassification rate of LDA in this case

LDA_misclass_rate <- nrow(test_sample_df[test_sample_df$y != 
                                           test_sample_df$LDA_predicted_y,]) /
  nrow(test_sample_df)

LDA_misclass_rate * 100

# Again, not perfect! What does the Bayes classification error look like?

bayes_misclass_rate <- nrow(test_sample_df[test_sample_df$y != 
                                             test_sample_df$bayes_predicted_y,]) /
  nrow(test_sample_df)

bayes_misclass_rate * 100

# LDA performs well - how close is it to the lowest possible misclassification
# error, as provided by bayes?

(LDA_misclass_rate - bayes_misclass_rate) * 100