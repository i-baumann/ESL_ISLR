# Logistic regression with a binary response and one predictor

# Load packages and read data
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse)

#####
# Step 1: Data generation

# Generate sample data for two outcomes (0 and 1). We will draw our x values
# from two normal distributions with slightly different means to guarantee
# some overlap between those observations with y = 0 and y = 1

y <- append(rep(0, each = 500), 
            rep(1, each = 500))

means <- sample.int(5, 2, replace = FALSE)

train_x <- append(rnorm(500, mean = means[1], sd = 1), 
                  rnorm(500, mean = means[2], sd = 1))

test_x <- append(rnorm(500, mean = means[1], sd = 1), 
                 rnorm(500, mean = means[2], sd = 1))

# We need to include an intercept term, so we will augment x with a 1-vector to
# accommodate the intercept. This will be our design matrix, or essentially just
# a one-power Vandermonde matrix

train_x <- cbind(1, train_x)
test_x <- cbind(1, test_x)

#####
# Step 2: Visualize the log-likelihood surface

# Logistic regression, like regular regression, is simply an optimization
# problem, except in this case the function we are maximizing is the log-
# likelihood function
#
# l(b) = sum(y %*% t(b) %*% x - log(1 + exp(t(b) %*% x)))
#
# which is strictly concave, so it will have one maximum. We will have
# two beta coefficients (one for the intercept and one for our single
# independent variable) so we can create a contour plot of the log-likelihood
# function's surface across values of these two coefficients.

b0_vec <- seq(-5, 5, .25)
b1_vec <- seq(-5, 5, .25)

surface <- expand_grid(b0_vec, b1_vec)
train_data <- cbind(y, train_x)

log_like_interior <- function(data, b_0, b_1){
  
  b_vec <- c(b_0, b_1)
  
  data[1] %*% t(b_vec) %*% data[2:3] -
    log(1 + exp(t(b_vec) %*% data[2:3]))
  
}

log_like_sum <- function(b_vec){
  
  sum(apply(train_data, 1, log_like_interior, b_0 = b_vec[1], b_1 = b_vec[2]))
  
}

surface$l <- apply(surface, 1, log_like_sum)

likelihood_surface_plot <- ggplot(surface, aes(b0_vec, b1_vec, z = l)) +
  geom_contour_filled(bins = 20) +
  labs(title = "Surface Plot of Log-Likelihood Function",
       subtitle = "Evaluated Across Range of Beta Coefficients",
       x = "Intercept",
       y = "B1",
       fill = "Log-Likelihood")

likelihood_surface_plot

#####
# Step 2a: Newton-Raphson/iteratively re-weighted least squares algorithm

# This is the method described in Elements of Statistical Learning

# To calculate the regression coefficients for our logistic regression we need
# to chose the coefficients that maximize the log-likelihood function
#
# l(b) = sum(y %*% t(b) %*% x - log(1 + exp(t(b) %*% x)))
#
# and in order to do this we obviously take its derivative and set it equal to
# zero. This derivative is the score equation
#
# dl(b)/db = sum(x %*% (y - p)) = 0
#
# where p is the fitted probabilities for each observation given our betas. We
# will use iteratively re-weighted least squares (see ESL) to "update" our beta
# coefficients over a number of iterations until our score equation evaluates 
# within a tolerable distance from zero (which signifies that we have maximized 
# the log-likelihood well enough).

beta_vec_newton <- c(0,0)
max_check <- FALSE
iter_newton <- 0

while (max_check != TRUE) {
  
  p <- exp(train_x %*% beta_vec_newton) / 
    (1 + exp(train_x %*% beta_vec_newton))
  
  W <- diag(as.vector(p))
  
  z <- train_x %*% beta_vec_newton + solve(W) %*% (y - p)
  
  beta_vec_newton <- solve(t(train_x) %*% W %*% train_x) %*% 
    t(train_x) %*% W %*% z
  
  # Check that we are actually maximizing the likelihood function by checking
  # if the score function is evaluating to zero
  
  score <- t(train_x) %*% (y - p)
  score <- c(score[1], score[2])
  max_check <- all.equal(c(0,0), score, 
                         tolerance = .Machine$double.eps ^ 0.5)
  iter_newton <- iter_newton + 1
  
}

#####
# Step 2b: Modified algorithm

# W is a bit nasty to work with, so we can also use a modified algorithm that
# incorporates W more directly. See 
# http://personal.psu.edu/jol2/course/stat597e/notes2/logit.pdf.

beta_vec_mod <- c(0,0)
max_check <- FALSE
iter_mod <- 0

while (max_check != TRUE) {
  
  p <- exp(train_x %*% beta_vec_mod) / 
    (1 + exp(train_x %*% beta_vec_mod))
  
  # Use `sweep` to multiply each row of x by scalar vector
  
  x_tilde <- sweep(train_x, MARGIN = 2, p * (1 - p), `*`)
  
  beta_vec_mod <- beta_vec_mod + solve(t(train_x) %*% x_tilde) %*%
    t(train_x) %*%
    (y - p)
  
  # Check that we are actually maximizing the likelihood function by checking
  # if the score function is evaluating to zero
  
  score <- t(train_x) %*% (y - p)
  score <- c(score[1], score[2])
  max_check <- all.equal(c(0,0), score, 
                         tolerance = .Machine$double.eps ^ 0.5)
  iter_mod <- iter_mod + 1
  
}

# The modified algorithm converges much more quickly than Newton-Raphson/IRLS:

iter_newton
iter_mod

# Does it appear as though these coefficients maximize the log-likelihood?

likelihood_surface_plot <- likelihood_surface_plot +
  geom_point(aes(x = beta_vec_mod[1], y = beta_vec_mod[2]))

likelihood_surface_plot

# Let's check our coefficients against those found by R's `glm` function

glm_model <- glm(y ~ train_x, data = train_data, family = "binomial")

glm_model$coefficients
beta_vec_newton
beta_vec_mod

#####
# 3: Visualize fit and test

# To plot the logistic fit we will use our coefficients to calculate the
# value predicted by the logistic coefficients and then plot those predictions
# against the actual values. For ease of plotting we will also convert the data
# matrix into a tibble, though this isn't strictly necessary.

train_data <- as_tibble(train_data)
colnames(train_data) <- c("y", "int", "x")

train_data$pred_prob <- exp(beta_vec_mod[1] + train_data$x * beta_vec_mod[2]) /
  (1 + exp(beta_vec_mod[1] + train_data$x * beta_vec_mod[2]))

logistic_plot <- ggplot(train_data) +
  geom_point(aes(x, y)) +
  geom_line(aes(x, pred_prob))

logistic_plot

# Let's also calculate the training error and include in the logistic fit plot
# which class our model assigns to the points

train_data$pred <- if_else(train_data$pred_prob < .5, 0, 1)

(nrow(train_data[(train_data$y != train_data$pred),]) / nrow(train_data)) * 100

logistic_plot_preds <- ggplot(train_data) +
  geom_point(aes(x, y, color = as.factor(pred))) +
  geom_line(aes(x, pred_prob)) +
  scale_color_manual(name = "Predicted y",
                     values = c("red", "blue"))

logistic_plot_preds

# To test the model, we'll use our coefficients from the model fit to the
# training data to predict outcome classifications for the test data: first we
# will calculate pr(Y = 1| X = x) for each observation using the training fit, 
# then assign Y = 0 if the predicted probability is less than .5 and Y = 1 if it 
# is greater than .5

test_data <- as_tibble(cbind(y, test_x))
colnames(test_data) <- c("y", "int", "x")

test_data$pred_prob <- exp(beta_vec_mod[1] + test_data$x * beta_vec_mod[2]) /
  (1 + exp(beta_vec_mod[1] + test_data$x * beta_vec_mod[2]))

test_data$pred <- if_else(test_data$pred_prob < .5, 0, 1)

# Calculate the test error

(nrow(test_data[(test_data$y != test_data$pred),]) / nrow(test_data)) * 100