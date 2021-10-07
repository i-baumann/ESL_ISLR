# Logistic Regression





This example walks through fitting a univariate logistic regression model to (idealized) generated data. It should be easy to see how this generalizes to multivariate cases. It assumes some prior knowledge of [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression){target="_blank"} and [log-odds](https://en.wikipedia.org/wiki/Logit){target="_blank"} (or see [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf#%5B%7B%22num%22%3A212%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D){target="_blank"}) and will be light on theory.

We'll skip doing a train/test split and comparison in this example and focus solely on implementing logistic regression from scratch.

## Data Generation

This example uses the [`tidyverse`](https://cran.r-project.org/web/packages/tidyverse/index.html){target="_blank"} and [`latex2exp`](https://cran.r-project.org/web/packages/latex2exp/latex2exp.pdf){target="_blank"} packages.


```r
library(tidyverse)
```

```
## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.1 ──
```

```
## ✓ ggplot2 3.3.5     ✓ purrr   0.3.4
## ✓ tibble  3.1.5     ✓ dplyr   1.0.7
## ✓ tidyr   1.1.4     ✓ stringr 1.4.0
## ✓ readr   1.4.0     ✓ forcats 0.5.1
```

```
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
```

```r
library(latex2exp)
```

We'll generate data with 1,000 observations from a simple case: a binary (0/1) balanced outcome...


```r
y <- append(rep(0, each = 500), 
            rep(1, each = 500))
```

...and a Gaussian independent variable. To ensure some overlap between the $y = 0$ and $y = 1$ observations we'll draw from two normal distributions with offset means. To include an intercept term we'll augment our independent variable vectors with a one-vector.


```r
means <- sample.int(6, 2, replace = FALSE)

x <- append(rnorm(500, mean = means[1], sd = 1),
                  rnorm(500, mean = means[2], sd = 1))

x <- cbind(1, x)

colnames(x) <- c("Int", "X1")
```

## The Log-Likelihood Surface

Like linear regression, logistic regression is simply an optimization problem. In the logistic case, the function we maximize is the [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function){target="_blank"} function

<center>
$$\ell (\vec{\beta}) = \sum_{i = 1}^N \left[ \vec{y}_i \vec{\beta}^T \vec{x}_i - \log(1 + e^{\vec{\beta}^T \vec{x}_i}) \right]$$
</center>

This function is strictly concave, so it will have one maximum and our $\beta$s ($\beta_0$ for the intercept and $\beta_1$ for our single independent variable) will (hopefully!) maximize this function. 

In this case we can easily visualize a log-likelihood surface by evaluating the function over some set of values for our two $\beta$s. First we'll create our set of $\beta$ values and then use those vectors to create a tibble of coordinates for our plot as well as put our $y$ vector and [design matrix](https://en.wikipedia.org/wiki/Design_matrix){target="_blank"} together to form a data matrix for easier evaluation.


```r
b0_vec <- seq(-20, 20, .5)
b1_vec <- seq(-20, 20, .5)

surface <- expand_grid(b0_vec, b1_vec)
data_matrix <- cbind(y, x)
```

Since we're evaluating the log-likelihood function over the data for all combinations of $\beta$ values in our set, we'll write two functions. The first handles the interior of the sum of the log-likelihood


```r
log_like_interior <- function(data, b_0, b_1){
  
  b_vec <- c(b_0, b_1)
  
  data[1] %*% t(b_vec) %*% data[2:3] -
    log(1 + exp(t(b_vec) %*% data[2:3]))

}
```

and the second handles the summation, [`apply`](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/apply){target="_blank"}ing over the data matrix.


```r
log_like_sum <- function(b_vec){
  
  sum(apply(data_matrix, 1, log_like_interior, b_0 = b_vec[1], b_1 = b_vec[2]))
  
}
```

Then we can `apply` the summation function over the $\beta$-value coordinated in our surface tibble and add the appropriate log-likelihood to that tibble.


```r
surface$l <- apply(surface, 1, log_like_sum)
```

Now that we have our $\beta$ coordinates and their corresponding log-likelihoods for our data, we can plot the log-likelihood surface.


```r
likelihood_surface_plot <- ggplot(surface, aes(x = b0_vec, y = b1_vec, z = l, fill = l)) +
  geom_raster(interpolate = TRUE) +
  geom_contour(bins = 50, color = "grey", size = 0.5) +
  scale_fill_distiller(palette = "YlOrRd", name = "Log-Likelihood", direction = 1) +
  labs(title = "Surface Plot of Log-Likelihood Function",
       subtitle = "Evaluated Across Range of Beta Coefficients",
       x = TeX("$\\beta_0$"),
       y = TeX("$\\beta_1$")) # +
  # theme(legend.position = "none")
```

<img src="binary_logistic_files/figure-html/unnamed-chunk-9-1.png" width="672" />

Our $\hat{\beta}_0$ and $\hat{\beta}_1$ coordinates estimated by our logistic regression should be in the bright yellow area, representing our highest-valued bin of log-likelihoods, at the maximum of the log-likelihood surface.

## Implementation

To fit the logistic regression model to the data and find the optimal $\hat{\beta}$ vector we set the derivatives of the log-likelihood function (or [score equations](https://en.wikipedia.org/wiki/Score_(statistics)){target="_blank"} ) with respect to $\vec{\beta}$ to zero:

<center>
$$\frac{\partial \ell(\vec{\beta})}{\partial \vec{\beta}} = \sum_{i = 1}^N x_i (y_i - \Pr (Y = 1 | X = x_i; \vec{\beta})) = 0$$
</center>

### Method 1: Newton-Raphson/IRLS

The method Hastie et al. use in [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf#%5B%7B%22num%22%3A202%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D){target="_blank"} is the [Newton-Raphson method](https://en.wikipedia.org/wiki/Newton%27s_method){target="_blank"} (which is re-written to be a [iteratively re-weighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares){target="_blank"} problem), which iteratively updates $\vec{\hat{\beta}}$ to find the maximum likelihood. We will use a stopping criteria that checks how the score equations for $\vec{\hat{\beta}}_0$ and $\vec{\hat{\beta}}_1$ evaluate and stops iterating when they are within a [tolerable distance from zero](https://en.wikipedia.org/wiki/Machine_epsilon){target="_blank"}.

This method uses a diagonal matrix $W$ which contains the weights derived from $\Pr (Y = 1 | X = x_i; \vec{\hat{\beta}})$ where $\vec{\hat{\beta}}$ contains the estimated coefficients from the previous iteration. These weights are then used to solve a [weighted least squares](https://en.wikipedia.org/wiki/Weighted_least_squares){target="_blank"} problem, which produces a vector $z$ of "adjusted" responses that are used to update the coefficients.

In each iteration $i$ we will solve

<center>
$$\vec{\hat{\beta}} \vphantom{1}^{i} = (X^T W X)^{-1} X^T W \vec{z}$$
</center>

where

<center>
$$z = X \vec{\hat{\beta}} \vphantom{1}^{i - 1} + W^{-1} (\vec{y} - \vec{p})$$
</center>

and $\vec{p}$ are the fitted probabilities given $\vec{\hat{\beta}} \vphantom{1}^{i - 1}$.


```r
beta_vec_irls <- c(0,0)
max_check <- FALSE
irls_iterations <- 0

while (max_check != TRUE) {
  
  p <- exp(x %*% beta_vec_irls) / 
    (1 + exp(x %*% beta_vec_irls))

  W <- diag(as.vector(p * (1 - p)))
  
  z <- x %*% beta_vec_irls + solve(W) %*% (y - p)
  
  beta_vec_irls <- solve(t(x) %*% W %*% x) %*% 
    t(x) %*% W %*% z
  
  # Check that we are actually maximizing the likelihood function by checking
  # if the score functions are evaluating to zero
  
  score <- t(x) %*% (y - p)
  score <- c(score[1], score[2])
  max_check <- all.equal(c(0,0), score, 
                         tolerance = .Machine$double.eps ^ 0.5)
  
  irls_iterations <- irls_iterations + 1
  
}
```

Our $\vec{\hat{\beta}}$ for this example from IRLS is:


```
##          [,1]
## Int 12.925375
## X1  -2.874321
```

### Method 2: Modified IRLS

Because $W$ is a $N \times N$ matrix, IRLS process may not always be effcient. We can side-step the large $W$ matrix and matrix operations involving it by instead directly multiplying the rows of the $X$ matrix by the predicted probability of $Y = 1$ for the observation corresponding to that row (given the "old" $\beta$ from the previous iteration); see [pseudo-code](http://personal.psu.edu/jol2/course/stat597e/notes2/logit.pdf#page=23){target="_blank"}. This new matrix will be $\tilde{X}$.

Note that, in R, `%*%` is a [`dot product`](https://en.wikipedia.org/wiki/Dot_product){target="_blank"} operator. But if two matrices have identical dimensions then the traditional multiplication operator `*` will perform an element-for-element scalar multiplication. So we will use our vector of probabilities to create an appropriately dimensioned matrix using [`matrix`](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/matrix){target="_blank"} and [`ncol`](https://www.rdocumentation.org/packages/hyperSpec/versions/0.98-20140523/topics/ncol){target="_blank"} to create $\tilde{X}$.


```r
beta_vec_mod <- c(0,0)
max_check <- FALSE
mod_iterations <- 0

while (max_check != TRUE) {
  
  p <- exp(x %*% beta_vec_mod) / 
    (1 + exp(x %*% beta_vec_mod))
  
  p_mat <- matrix(p * (1 - p), length(p * (1 - p)), ncol(x))
  
  # Use `sweep` to multiply each row of x by scalar vector
  
  x_tilde <- x * p_mat
  
  beta_vec_mod <- beta_vec_mod + solve(t(x) %*% x_tilde) %*%
    t(x) %*%
    (y - p)
  
  # Check that we are actually maximizing the likelihood function by checking
  # if the score functions are evaluating to zero
  
  score <- t(x) %*% (y - p)
  score <- c(score[1], score[2])
  
  max_check <- all.equal(c(0,0), score, 
                         tolerance = .Machine$double.eps ^ 0.5)
  
  mod_iterations <- mod_iterations + 1
  
}
```

This will sometimes converge a little more quickly. The number of iterations for this particular example were


```
## [1] "IRLS iterations: 9"
```

```
## [1] "Modified IRLS iterations: 9"
```

Our $\vec{\hat{\beta}}$ for this example from the modified algorithm is:


```
##          [,1]
## Int 12.925375
## X1  -2.874321
```

## Testing

### Comparing to `glm`

We can easily check the coefficients we've calculated against those produced by R's `glm`. By default `glm` [uses IRLS](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/glm){target="_blank"} to fit models.


```r
glm_model <- glm(y ~ X1, data = as.data.frame(data_matrix), family = "binomial")
```


```r
beta_vec_irls
```

```
##          [,1]
## Int 12.925375
## X1  -2.874321
```

```r
beta_vec_mod
```

```
##          [,1]
## Int 12.925375
## X1  -2.874321
```

```r
glm_model$coefficients
```

```
## (Intercept)          X1 
##   12.925375   -2.874321
```

### The Log-Likelihood Surface

We can also use the log-likelihood surface plot to (approximately) verify that the coefficients we estimated are correct and that the logistic regression estimates do maximize the log-likelihood function by plotting the point for our estimates on the surface plot:


```r
likelihood_surface_plot <- likelihood_surface_plot +
  geom_point(aes(x = beta_vec_mod[1], y = beta_vec_mod[2]))
```

<img src="binary_logistic_files/figure-html/unnamed-chunk-18-1.png" width="672" />

### Plotting the Logistic Fit

To get the classic logistic regression plot we can calculate

<center>
$$\Pr (Y = 1 | X = x_i; \vec{\beta}) = \frac{e^{\vec{x} \hat{\vec{\beta^T}}}}{1 + e^{\vec{x} \hat{\vec{\beta^T}}}}$$
</center>

for all $x_i$ and plot those probabilities as a line against the data points on the $(x, y)$ plane. We'll also color-code the points by how our model fit classifies the points, using the decision rule that an observation is classified to $\hat{Y} = 0$ if $\Pr (Y = 1 | X = x_i; \vec{\beta}) < 0.5$ and $\hat{Y} = 1$ otherwise.


```r
pred_prob <- exp(data_matrix[, 2:3] %*% beta_vec_mod) / 
    (1 + exp(data_matrix[, 2:3] %*% beta_vec_mod))

data_tibble <- as_tibble(data_matrix)

data_tibble$pred_prob <- pred_prob[,1]

data_tibble$pred <- if_else(data_tibble$pred_prob < .5, 0, 1)

logistic_plot_preds <- ggplot(data_tibble) +
  geom_point(aes(X1, y, color = as.factor(pred))) +
  geom_line(aes(X1, pred_prob)) +
  scale_color_manual(name = "Predicted y",
                      values = c("red", "blue"))
```

<img src="binary_logistic_files/figure-html/unnamed-chunk-20-1.png" width="672" />