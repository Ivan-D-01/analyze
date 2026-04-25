Regularization
Problems
1. Fit Ridge and LASSO regressions to predict both Рас and categorical variables.
Get the optimal regularization parameter by cross-validation. Are there differences in
precision? Compare that to the simple (without regularization) model.
2. Let Xn = (xn,1, . . . , xn,M )T , where xn,m ∼ N (0, 1) for n = 1, . . . , N . Let ε ∼ N (0, 1)
as well for n = 1, . . . , N . Let yn = xn,1 − 2xn,2 + εn. Generate such dataset for any
N ≥ 20, M ≥ 5 of your choice. Fit LASSO regression with the optimal regularization
parameter (chosen by cross-validation). Did it catch the true parameters? Compare with
least squares and ridge regressions.
3. For the dataset from the previous problem, apply some stepwise variable selection
algorithm. Did it do a better job than LASSO?

Recommendations
To perform Ridge and LASSO regularization, you can use glmnet package, function glmnet.
Cross-validation can be calculated using cv.glmnet function. Keep in mind that sometimes the
auto-generated parameter lambda for these functions is not adequate, so you need to provide
the sequence yourself.
To generate normal random variables, you need to use rnorm function. To create a matrix,
there is a matrix function.
The stepwise selection algorithms are available in step function from stats package. But the
only optimization criteria options are AIC and BIC. If you want to realize the cross-validation,
then do it manually or use some package like SuperLearner