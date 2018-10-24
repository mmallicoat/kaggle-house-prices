Todo
----

*   Implement a learning curve to see whether the model is over-
    or under-fitting
*   If over-fitting, add regularization. If under-fitting, try
    adding more features.
*   To determine which features to add, maybe try using decision
    trees / random forest? Or look a correlations between indep.
    variables and the response variable?
*   Implement preprocessing and variable transformations in a
    model elegant and repeatable way

Done
````
*   Restructure model to separate data prep from training,
    and training from testing a model.

Preproccessing
--------------

It seems that ``sklearn`` may not have the ability to use a
lognormal response in a linear model. To implement this, I may
need to do some power transform (such as Box–Cox) in a
preprocessing step. See `sklearn docs on preprocessing
<http://scikit-learn.org/stable/modules/preprocessing.html>`__.

The scalers includej

`This article
<https://ryankresse.com/convenient-preprocessing-with-sklearn_pandas-dataframemapper/>`__
shows how the sklearn ``DataFrameMapper`` can store and enact
transformations of a dataframe (such as normalizing variables).
This makes it easy to repeat the exact same transformations on the
training, CV, and test data.  **The mapper can also be pickled for
later use.** This mapper can also be used to encode
categorical variables, using ``OneHotEncoder`` or
``OrdinalEncoder``.

sklearn has `pipelines
<http://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline>`__,
too.
