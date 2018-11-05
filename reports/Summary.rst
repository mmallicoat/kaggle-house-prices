Predicting House Prices
=======================

Intro
-----

The (data science) company Kaggle `ran a competition
<https://www.kaggle.com/c/house-prices-advanced-regression-techniq
ues/data>`__ (?) where the task was to predict the price of a
house given a large number of measurements of the house: the area
of the plot, the floor area, the number of bedrooms, etc. The data
are from houses in Ames, Iowa, compiled for use in data science
education. To solve the problem posed, I decided to develop a
simple Generalized Linear Regression (GLM) model.

Data Prep
---------

The first step I took was to stratify the data provided,
complete with the price of each house, into training and
cross-validation (CV) datasets. Having a CV dataset lets you
compare the performance of models on out-of-sample data, thereby
avoiding overfitting and getting a more accurate estimate of its
perfomrance. I randomly partitioned the labeled dataset into the
training and CV datasets, since I do not know if there is some
order to how they are presented in the file Kaggle provides which
might bias my model.

Next, I dealt with any missing values in the dataset. For numeric
variables, I calculated the mean of the variable in the training
data and substituted this value for any missing in the training
and CV datasets. For the categorical variables, I substituted a
new value "Unknown."

To select from the many features, I used some simple heuristics.
For the numeric variables, I calculated the variance of each
within the training data. I plotted a histogram of these and
found an "elbow" where there was a significant drop-off in
variance. I then selected all variables with variance above this
threshold. This amounted to about 25% (?) of the variables. The
rationale behind this is that variables with low variance do not
provide much discriminating information between houses, since
all of the houses will have similar values. For each of the
categorical variables, I calculated the entropy_, assuming each
value was pulled from a multinomial distribution. Entropy is a
measure of the amount of "information" contained in a stoachastic
process. Random variables with little "surprise" in their realized
values will have low entropy. For binary variables, the entropy
calculated is equivalent to the variance of the corresponding
Bernoulli distribution. After calculating the entropy, I plotted a
histogram, found an "elbow" is use for the threshold, and selected
all of the variables above this threshold, in the same manner
as the numeric variables. This amounted to about 25% (?) of the
categorical variables.

[Show plot of elbow in variance?]

.. _entropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)

The selected categorical variables were then encoded as dummy
variables, [#]_ so that they can be included in the regression.
One trip-up was that there are values of categorical variables
appearing in the test data that do not appear in the training/CV
data. The universe of values must be known ahead of time in order
to encode the variable as dummy variables. Given a new dataset
containing unseen values of a categorical variable, the model
could not be applied.

.. [#] Dummy variables are a collection of binary variables whose
    combination correspond to one of the values of the categorical
    variable. The simplest example is a variable with possible values
    "Male" and "Female" being encoded as 1 and 0. For variables with
    *n* possible values, *n - 1* dummy variables are required.


.. TODO: rewrite follow paragraph.

Challenges: developing a pipeline that can be applied to all datasets uniformly. This is especially a problem when the procedure is "fitted": when the function of a the training data. These parameters must then be stored somewhere so they can be applied to the CV and testing datasets. Instead of writing out the parameters, as I did for the scaler and regression parameters, I simply kept them in memory and processed all of the datasets at once. This is less than ideal, since I wouldn't have the training parameters readily available to apply to a new dataset. These would be needed to put such an application into production.

Training
--------

Before training the model, I performed some transformations on
the data. I standardized the features, subtracting the mean and
dividing by standard deviation to create features with zero mean
and unit variance.

Also, I log-transformed the response variable for two reasons.
First, house prices are not normally distribution, like most
currency values, which violates an assumption of the linear
regression. By log-transforming the response variable, it is
"closer" to following a normal distribution. [#]_

.. TODO: insert histogram chart; maybe overlay bell curve

Additionally, the loss function specified for the Kaggle
competition is the mean squared error of the *logarithm* of
the house prices predicted. If we wish to develop a model that
performs well under this loss function, we must optimize the
parameters of our model with respect to it.

.. [#] Processes that are the sum of many independent occurrences
    generally follow a normal distribution, which is consistence with
    the Central Limit Theorem. An example of this is human height,
    which is perhaps the result of the expression of many different
    genes, the quality of nutrition through each phase of childhood,
    the effects of childhood disease, etc., which are generally
    independence events, each having a small effect. Processes like
    prices or salaries cannot be normally distributed on the face
    since they cannot have negative values. Secondly, instead of the
    constituents having an additive effect, they seem to have more of
    a *multiplicative* effect on the outcome. Learning two new skills
    will increase your salary more than that sum of each alone.

First, I tried using a GLM with a log response, but this resulted in
some very large positive and negative coefficients in the fitted
model. These coefficients lead to some overflows and underflows_,
respectively, in the predicted value of the response variable, due
to the limited precision in floating point arithmetic. To remedy
this, I used a ridge regression instead, which adds regularization
thereby penalizing coefficients with large magnitude. This solved
the problem.

.. _underflows: https://en.wikipedia.org/wiki/Arithmetic_underflow

After training the model, I saved the parameters of both the
standardization procedure and the linear regression. These are
both are needed in order to repeat the preprocessing steps and
make predictions from the CV and test datasets.

.. TODO: rewrite following two paragraphs

Challenges: Selecting features becomes part of the model, but
these procedures are not present in the parameters of the model
(e.g., the coefficients and intercept of a linear regression).
The entire pipeline from raw data to the final model has to be
preserved. If you want to prototype multiple models (say I wanted
to try a random forest model as well), you must keep multiple
versions of this pipeline, even though there is a large amount of
code re-use. On the other hand, by writing code in a general way
to make it easier to re-use, it may take a longer time and be less
convenient to use for any one model.

I followed the 'directed acyclic graph' guidance from "Cookiecutter Data Science." To follow these with multiple protype models, you would need multiple Makefiles, perhaps. Thus guidance might be good for the published project, but may be less convenient when prototyping.

Prediction
----------

To make predictions given the CV and test datasets, the
preprocessing steps and repeated:

1. Standardize the variables using means and standard deviations
   from training dataset
2. For the CV dataset, log-transform the response variable. (We do
   not know the value of the response variable for the testing data,
   of course.)
3. Apply our regression model to make a prediction: multiply
   values of the features by the fitted coefficients, sum these up,
   and add the intercept.
4. For the CV dataset, calculate the value of the loss function as
   a diagnostic.
5. Before writing out the predictions, reverse the log-transform
   by exponentiating the predicted value.

Compared to the leaderboard on the Kaggle website, my model's performance in fairly middling. There are many optimiziations left to be made. Here are some things to try in the future:

*   Engineer some custom features, especially ones that capture
    interactions between variables. These might be something like the
    ratio of bathrooms to bedrooms, or ratio of plot area to house
    floor area.
*   Make use of the ordinal variables: there are some variables that
    are actually ordinal, not categorical. An example of this is X.
    Instead of ignoring the ordering of the levels of the variable,
    they could be taken advantage of.
*   Try some alternate models, especially those that can fit
    non-linear functions. There may be some non-linear interactions
    between the house price and the indepdenent vairables, such as
    the price not being monotonically increasing with the value of an
    independence variable. One plausible explanation of this might be
    something along the lines of: a larger yard may correlate with a
    more valuable property, but it may correlate with a more rural
    location; the negative effect of the rural location on the house
    price might outweight the increase from the larger yard.
*   Supplement external data: we are given the names of
    neighborhoods of the houses. There is publically available data on
    houses and their prices from these locations. This data could be
    collected and used to supplement the data provided by Kaggle. Or,
    a secondary model could be built from the external data and then
    combined with the model trained on the Kaggle data in an ensemble.
