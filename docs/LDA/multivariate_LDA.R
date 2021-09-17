# Linear discriminant analysis in a multivariate setting with three classes

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

# Note that LDA *always* assumes each class has the same covariance matrix, so
# this is an idealized example

pop_corr <- runif(1, 0, 1)
pop_var <- runif(1, 0, 10)
pop_sigma <- matrix(c(pop_var, pop_corr,
                      pop_corr, pop_var), 2, 2)

# Generate sample data from multivariate normal distributions, choosing n samples

n <- 300

c0_train <- mvrnorm(n = n,
              mu = mu_c0,
              Sigma = pop_sigma)

c1_train <- mvrnorm(n = n,
              mu = mu_c1,
              Sigma = pop_sigma)

c2_train <- mvrnorm(n = n,
              mu = mu_c2,
              Sigma = pop_sigma)

c0_test <- mvrnorm(n = n,
              mu = mu_c0,
              Sigma = pop_sigma)

c1_test <- mvrnorm(n = n,
              mu = mu_c1,
              Sigma = pop_sigma)

c2_test <- mvrnorm(n = n,
              mu = mu_c2,
              Sigma = pop_sigma)

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
  t(x_vec) %*% solve(pop_sigma) %*% pop_c0_mean_vec -
    .5 * t(pop_c0_mean_vec) %*% solve(pop_sigma) %*% pop_c0_mean_vec +
    log(c0_prior)
}

d1_bayes <- function(x_vec){
  t(x_vec) %*% solve(pop_sigma) %*% pop_c1_mean_vec -
    .5 * t(pop_c1_mean_vec) %*% solve(pop_sigma) %*% pop_c1_mean_vec +
    log(c1_prior)
}

d2_bayes <- function(x_vec){
  t(x_vec) %*% solve(pop_sigma) %*% pop_c2_mean_vec -
    .5 * t(pop_c2_mean_vec) %*% solve(pop_sigma) %*% pop_c2_mean_vec +
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

# Like in the univariate case, we can plot the Bayes decision boundaries: the
# sets of x's for which the scores between classes are equal. All lines will
# intersect each other at one common point. The lines will be orthogonal to
# S^{-1} (m_i - m_j) where S is the population covariance matrix (which is 
# common to all classes, remember) and m is the class-level population 
# mean-vector of X1 and X2 for classes i and j. These lines will also contain 
# the midpoints between the mean-vectors for each class.

# See this excellent stack overflow post for a thorough and intuitive answer 
# using the famous iris dataset:
# https://stats.stackexchange.com/questions/92157/compute-and-graph-the-lda-decision-boundary

# Calculate midpoints between class-level population mean-vectors

pop_c0_c1_midpoint <- (pop_c0_mean_vec + pop_c1_mean_vec) / 2
pop_c0_c2_midpoint <- (pop_c0_mean_vec + pop_c2_mean_vec) / 2
pop_c1_c2_midpoint <- (pop_c1_mean_vec + pop_c2_mean_vec) / 2

# Generate the population orthogonal vectors

pop_c0_c1_ortho <- Null(solve(pop_sigma) %*% 
                      (pop_c0_mean_vec - pop_c1_mean_vec))
pop_c0_c2_ortho <- Null(solve(pop_sigma) %*% 
                      (pop_c0_mean_vec - pop_c2_mean_vec))
pop_c1_c2_ortho <- Null(solve(pop_sigma) %*% 
                      (pop_c1_mean_vec - pop_c2_mean_vec))

# Plot the decision boundaries using the midpoints between class means and the
# vectors orthogonal to S^{-1} (m_i - m_j)

train_sample_scatter <- train_sample_scatter +
  geom_segment(aes(x = pop_c0_c1_midpoint[1] - 3 * pop_c0_c1_ortho[1],
                   xend = pop_c0_c1_midpoint[1] + 3 * pop_c0_c1_ortho[1],
                   y = pop_c0_c1_midpoint[2] - 3 * pop_c0_c1_ortho[2],
                   yend = pop_c0_c1_midpoint[2] + 3 * pop_c0_c1_ortho[2]),
               linetype = "solid") +
  geom_segment(aes(x = pop_c0_c2_midpoint[1] - 3 * pop_c0_c2_ortho[1],
                   xend = pop_c0_c2_midpoint[1] + 3 * pop_c0_c2_ortho[1],
                   y = pop_c0_c2_midpoint[2] - 3 * pop_c0_c2_ortho[2],
                   yend = pop_c0_c2_midpoint[2] + 3 * pop_c0_c2_ortho[2]),
               linetype = "solid") +
  geom_segment(aes(x = pop_c1_c2_midpoint[1] - 3 * pop_c1_c2_ortho[1],
                   xend = pop_c1_c2_midpoint[1] + 3 * pop_c1_c2_ortho[1],
                   y = pop_c1_c2_midpoint[2] - 3 * pop_c1_c2_ortho[2],
                   yend = pop_c1_c2_midpoint[2] + 3 * pop_c1_c2_ortho[2]),
               linetype = "solid")

train_sample_scatter

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

# Since we cannot know the population parameters and covariance matrix,
# we will also build a class-weighted covariance matrix from the sample data. 
# The sample covariance matrix is constructed by de-meaning the X1 and X2 
# vectors by the populations means of the classes corresponding to the 
# individual observations, then each observation-wise (X1, X2) vector is 
# multiplied by its transpose. All of these observation-level covariance 
# matrices are summed to produce the sample covariance matrix, which is 
# corrected by dividing by (N - K) where N is the total number of observations 
# in the data and K is the number of classes. This is similar to Bessel's 
#,correction: https://en.wikipedia.org/wiki/Bessel%27s_correction. These lines 
# will also contain the midpoints between the mean-vectors for each class.

# De-mean the X1 and X2 vectors class-wise using the sample class-wise X1
# and X2 means

sample_mean_c0_X1 <- mean(train_sample_df$X1[train_sample_df$y == 0])
sample_mean_c0_X2 <- mean(train_sample_df$X2[train_sample_df$y == 0])
sample_mean_c1_X1 <- mean(train_sample_df$X1[train_sample_df$y == 1])
sample_mean_c1_X2 <- mean(train_sample_df$X2[train_sample_df$y == 1])
sample_mean_c2_X1 <- mean(train_sample_df$X1[train_sample_df$y == 2])
sample_mean_c2_X2 <- mean(train_sample_df$X2[train_sample_df$y == 2])

train_sample_df <- train_sample_df %>%
  mutate(centered_X1_sample = if_else(y == 0, X1 - sample_mean_c0_X1,
                                   if_else(y == 1, X1 - sample_mean_c1_X1,
                                           X1 - sample_mean_c2_X1)),
         centered_X2_sample = if_else(y == 0, X2 - sample_mean_c0_X2,
                                   if_else(y == 1, X2 - sample_mean_c1_X2,
                                           X2 - sample_mean_c2_X2)))

# Build the sample covariance matrix

train_sigma_LDA <- matrix(c(0, 0, 0, 0), 2, 2)

for (i in 1:nrow(train_sample_df)) {
  
  temp_cov <- c(train_sample_df$centered_X1_sample[i], train_sample_df$centered_X2_sample[i])
  
  train_sigma_LDA <- train_sigma_LDA + temp_cov %*% t(temp_cov)
  
}

# Correct the covariance matrix

train_sigma_LDA <- train_sigma_LDA - (n * 3 - 3)

# Calculate midpoints between class-level sample mean-vectors

sample_c0_c1_midpoint <- (sample_c0_mean_vec + sample_c1_mean_vec) / 2
sample_c0_c2_midpoint <- (sample_c0_mean_vec + sample_c2_mean_vec) / 2
sample_c1_c2_midpoint <- (sample_c1_mean_vec + sample_c2_mean_vec) / 2

# Create the LDA classifier

d0_LDA <- function(x_vec){
  t(x_vec) %*% solve(train_sigma_LDA) %*% sample_c0_mean_vec -
    .5 * t(sample_c0_mean_vec) %*% solve(train_sigma_LDA) %*% 
    sample_c0_mean_vec + log(c0_prior)
}

d1_LDA <- function(x_vec){
  t(x_vec) %*% solve(train_sigma_LDA) %*% sample_c1_mean_vec -
    .5 * t(sample_c1_mean_vec) %*% solve(train_sigma_LDA) %*% 
    sample_c1_mean_vec + log(c1_prior)
}

d2_LDA <- function(x_vec){
  t(x_vec) %*% solve(train_sigma_LDA) %*% sample_c2_mean_vec -
    .5 * t(sample_c2_mean_vec) %*% solve(train_sigma_LDA) %*% 
    sample_c2_mean_vec + log(c2_prior)
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

# Apply the LDA classifier to the training data

train_sample_df$LDA_predicted_y <- apply(train_sample_df[, c("X1", "X2")], 
                                           1, LDA_classifier)

# Generate the sample orthogonal vectors for plotting

sample_c0_c1_ortho <- Null(solve(train_sigma_LDA) %*% 
                             (sample_c0_mean_vec - sample_c1_mean_vec))
sample_c0_c2_ortho <- Null(solve(train_sigma_LDA) %*% 
                             (sample_c0_mean_vec - sample_c2_mean_vec))
sample_c1_c2_ortho <- Null(solve(train_sigma_LDA) %*% 
                             (sample_c1_mean_vec - sample_c2_mean_vec))

# Plot the LDA decision boundaries (as dashed lines, with the optimal Bayes
# boundaries in solid lines)

train_sample_scatter <- train_sample_scatter +
  geom_segment(aes(x = sample_c0_c1_midpoint[1] - 3 * sample_c0_c1_ortho[1],
                   xend = sample_c0_c1_midpoint[1] + 3 * sample_c0_c1_ortho[1],
                   y = sample_c0_c1_midpoint[2] - 3 * sample_c0_c1_ortho[2],
                   yend = sample_c0_c1_midpoint[2] + 3 * sample_c0_c1_ortho[2]),
               linetype = "dashed") +
  geom_segment(aes(x = sample_c0_c2_midpoint[1] - 3 * sample_c0_c2_ortho[1],
                   xend = sample_c0_c2_midpoint[1] + 3 * sample_c0_c2_ortho[1],
                   y = sample_c0_c2_midpoint[2] - 3 * sample_c0_c2_ortho[2],
                   yend = sample_c0_c2_midpoint[2] + 3 * sample_c0_c2_ortho[2]),
               linetype = "dashed") +
  geom_segment(aes(x = sample_c1_c2_midpoint[1] - 3 * sample_c1_c2_ortho[1],
                   xend = sample_c1_c2_midpoint[1] + 3 * sample_c1_c2_ortho[1],
                   y = sample_c1_c2_midpoint[2] - 3 * sample_c1_c2_ortho[2],
                   yend = sample_c1_c2_midpoint[2] + 3 * sample_c1_c2_ortho[2]),
               linetype = "dashed") +
  labs(caption = "Bayes classifier boundaries are solid; LDA boundaries are dashed")

train_sample_scatter

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

# How close is it to the lowest possible misclassification error, as provided by
# Bayes, for the test set?

(LDA_misclass_rate - bayes_misclass_rate) * 100