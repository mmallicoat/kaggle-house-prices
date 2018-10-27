Todo
----

*   Select features using variance threshold, possibly in
    make_features.py.
*   Re-predict

Won't Do
````````
*   Implement a learning curve to see whether the model is over-
    or under-fitting
*   If over-fitting, add regularization. If under-fitting, try
    adding more features.
*   Implement preprocessing and variable transformations in a
    model elegant and repeatable way

Done
````
*   Encode categorical variables.
*   Restructure model to separate data prep from training,
    and training from testing a model.

Feature Selection
-----------------

`sklearn article
<http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection>`__.

Types: correlation, scoring (?), wrapper.

`Pipeline and feature selection
<http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-as-part-of-a-pipeline>`__:
"Feature selection is usually used as a pre-processing step before
doing the actual learning. The recommended way to do this in
scikit-learn is to use a ``sklearn.pipeline.Pipeline``."

Preproccessing
--------------

It seems that ``sklearn`` may not have the ability to use a
lognormal response in a linear model. To implement this, I may
need to do some power transform (such as Box–Cox) in a
preprocessing step. See `sklearn docs on preprocessing
<http://scikit-learn.org/stable/modules/preprocessing.html>`__.

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
too. `More on pipelines
<http://scikit-learn.org/stable/modules/compose.html#pipeline>`__.

Article on the `difference between normalization, standardization,
and regularization
<https://maristie.com/blog/differences-between-normalization-standardization-and-regularization/>`__.
A `FAQ <http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html>`__ on the topic.
