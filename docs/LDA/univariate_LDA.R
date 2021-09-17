# Linear discriminant analysis in a univariate setting with two classes

library(tidyverse)

#####
# Step 1: Data generation

# Generate sample data for two outcome classes (0 and 1)
# These populations will have different means but equal variance, Gaussian
# in distribution, and will overlap some; we will have a training sample and
# a test sample

population0_mean <- 1
population0_sd <- 1

population1_mean <- 3
population1_sd <- 1

c0_train <- tibble(y = 0,
             x = rnorm(1000, 
                       mean = population0_mean, 
                       sd = population0_sd))
c1_train <- tibble(y = 1,
             x = rnorm(1000, 
                       mean = population1_mean, 
                       sd = population1_sd))

c0_test <- tibble(y = 0,
                   x = rnorm(1000, 
                             mean = population0_mean, 
                             sd = population0_sd))
c1_test <- tibble(y = 1,
                   x = rnorm(1000, 
                             mean = population1_mean, 
                             sd = population1_sd))

train_sample_df <- bind_rows(c0_train, c1_train)
test_sample_df <- bind_rows(c0_test, c1_test)

# Plot the distributions of the population data by class

population_density <- ggplot(data.frame(x = c(-4, 8)), aes(x)) + 
  stat_function(fun = dnorm, 
                aes(color = "Class: 0"),
                args = list(mean = population0_mean, sd = population0_sd)) +
  stat_function(fun = dnorm, 
                aes(color = "Class: 1"),
                args = list(mean = population1_mean, sd = population1_sd)) +
  scale_colour_manual(values = c("red", "blue")) +
  labs(title = "Population Densities of Two Classes") +
  theme(legend.title = element_blank())

population_density

# Plot the distributions of the training and test samples by class

train_sample_histogram <- ggplot(train_sample_df) +
  geom_histogram(aes(x = x, fill = as.factor(y)), 
                 alpha = .7,
                 position = "identity") +
  scale_fill_manual(values = c("red", "blue"),
                    labels = c("Class: 0", "Class: 1")) +
  labs(title = "Training Sample Histograms of Two Classes") +
  theme(legend.title = element_blank())

train_sample_histogram

test_sample_histogram <- ggplot(test_sample_df) +
  geom_histogram(aes(x = x, fill = as.factor(y)), 
                 alpha = .7,
                 position = "identity") +
  scale_fill_manual(values = c("red", "blue"),
                    labels = c("Class: 0", "Class: 1")) +
  labs(title = "Test Sample Histograms of Two Classes") +
  theme(legend.title = element_blank())

test_sample_histogram

#####
# Step 2: Calculate optimal Bayes decision boundary (for comparison)

# The Bayes classifier has the lowest classification error of any classifier
# and classifies observations based on the probability of being in a class
# given its predictor (x)

# See: https://en.wikipedia.org/wiki/Bayes_classifier

# The Bayes decision boundary is the boundary for which the probability of an
# observation being classified by the Bayes classifier is equal among classes;
# in this case, we will only have one boundary because we only have two classes

# We will compute the optimal Bayes decision boundary (using the population
# data) to compare our LDA against

bayes_decision <- (population0_mean ^ 2 - population1_mean ^ 2) /
  (2 * (population0_mean - population1_mean))

# Add the optimal Bayes decision boundary to the population density plot

population_density + geom_vline(xintercept = bayes_decision,
                                color = "black")

#####
# Step 3: Create the LDA classifier and train

# The LDA classifier uses estimates of mean and variance for each
# class as well as a discriminant function to determing the probability that
# an observation is of a particular class. The LDA classifier then assigns to
# each observation that class for which the probability of membership is highest

# The general LDA uses Bayes' theorem for continuous variables (see:
# https://en.wikipedia.org/wiki/Bayes%27_theorem#For_continuous_random_variables)
# where, since we have Gaussian data, we use the normal probability density
# function (for each class, respectively) as our f() function

# For our estimates of the mean and variance we use the empirical
# class-specific means and standard deviations (even though the class
# variances are equivalent in this case we will treat them separately for
# completeness) from our training samples. We also need to calculate the 
# empirical prior probability that an observation belongs to each class (in this
# case, both will be .5 since we have equal samples).

class_0_mean <- mean(train_sample_df$x[train_sample_df$y == 0])
class_0_var <- var(train_sample_df$x[train_sample_df$y == 0])

class_1_mean <- mean(train_sample_df$x[train_sample_df$y == 1])
class_1_var <- var(train_sample_df$x[train_sample_df$y == 1])

class_0_prior <- length(train_sample_df$x[train_sample_df$y == 0]) /
  length(train_sample_df$x)

class_1_prior <- length(train_sample_df$x[train_sample_df$y == 1]) /
  length(train_sample_df$x)

# Whereas in Bayes classification the classification is determined strictly by
# probability of class membership, in LDA we use the linear score function,
# which is the gradient of the log-likelihood function

# Since we have two classes, we will calculate the scores that each observation 
# belongs to each class in the training data

d0 <- function(x){
  x * (class_0_mean / class_0_var) - (class_0_mean ^ 2 / (2 * class_0_var ^ 2)) +
    log(class_0_prior)
}

d1 <- function(x){
  x * (class_1_mean / class_1_var) - (class_1_mean ^ 2 / (2 * class_1_var ^ 2)) +
    log(class_1_prior)
}

# The LDA classifier simply compares the scores for a given observation's
# predictor value and assigns the observation to the class for which its score
# is largest

LDA <- function(x){
  
  score_0 <- d0(x)
  score_1 <- d1(x)
  
  if (score_0 > score_1) {
    0
  } else {
    1
  }
}

# Apply the LDA classifier to the samples

train_sample_df$predicted_y <- lapply(train_sample_df$x, LDA)
train_sample_df$predicted_y <- as.integer(train_sample_df$predicted_y)

# Our LDA discriminant functions are linear in x, so we can solve for the point
# x that serves as the LDA decision boundary (where the scores are equal)

LDA_decision <- (class_0_mean ^ 2 / (2 * class_0_var) -
                   class_1_mean ^ 2 / (2 * class_0_var) +
                   log(class_1_prior) - log(class_0_prior)) /
                (class_0_mean / class_0_var - class_1_mean / class_1_var)

#####
# Step 4: Analyze performance

# Now that our LDA is trained and we have its decision boundary, apply it to the 
# test data

test_sample_df$predicted_y <- if_else(test_sample_df$x < LDA_decision,
                                             0, 1)

# Find the misclassification rate of LDA in this case

LDA_misclass_rate <- nrow(test_sample_df[test_sample_df$y != 
                                           test_sample_df$predicted_y,]) /
                     nrow(test_sample_df)

LDA_misclass_rate * 100

# It's not perfect! How does this compare to the Bayes classifier's performance?

test_sample_df$bayes_predicted_y <- if_else(test_sample_df$x < bayes_decision,
                                       0, 1)

bayes_misclass_rate <- nrow(test_sample_df[test_sample_df$y != 
                                             test_sample_df$bayes_predicted_y,]) /
                       nrow(test_sample_df)

bayes_misclass_rate * 100

# LDA performs well - how close is it to the lowest possible misclassification
# error, as provided by bayes?

(LDA_misclass_rate - bayes_misclass_rate) * 100

# Pretty close!

# Plot the Bayes decision boundary and the LDA decision boundary on the test
# sample histogram

test_sample_histogram + 
  geom_vline(xintercept = bayes_decision,
                                   color = "black") +
  geom_vline(xintercept = LDA_decision,
             color = "black",
             linetype = "longdash") +
  labs(subtitle = "Solid line: Bayes (optimal) decision boundary\nDashed line: LDA decision boundary")
  