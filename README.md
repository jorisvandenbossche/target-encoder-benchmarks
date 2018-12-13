# Target Encoding benchmarks

This repo contains a set of benchmarks comparing different Target Encoding
options (and comparing with a One Hot Encoder).

Target encoding is a method to encode a categorical variable in which each
category is encoded given the effect it has on the target variable y.
It can especially be useful for high-cardinality categorical data (for which
a one hot encoding would result in a high dimensionality).

References:

- *"A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems"* (Micci-Barreca, 2001): https://dl.acm.org/citation.cfm?id=507538

That paper describes an Empirical Bayes method to perform shrinkage of the
expected value of the target.\
Other algorithms to prevent overfitting are a leave one out method, where the target of the sample itself is not used in determining the expected value, adding noise, or determining the expected value in a cross-validation.

Currently available implemenations:

- The `category_encoders` package (https://github.com/scikit-learn-contrib/categorical-encoding)
  has a `TargetEncoder` and `LeaveOneOutEncoder` class.
- The `dirty_cat` package (https://github.com/dirty-cat/dirty_cat) also has a
  `TargetEncoder`.
- The `hccEncoding` project (https://hccencoding-project.readthedocs.io/en/latest/)
  has an implementation of the empirical Bayes and Leave One Out method, and for each
  a version without and with KFold cross-validation.


### Set-up

Needed packages:

```
conda install scikit-learn pandas matplotlib seaborn statsmodels xgboost lightgbm joblib
```

Additional packages I installed from git master:

```
cd repos

git clone https://github.com/Robin888/hccEncoding-project.git
cd hccEncoding-project
pip install -e .
cd ..

git clone https://github.com/dirty-cat/dirty_cat.git
cd dirty_cat
pip install -e .
cd ..

git clone https://github.com/scikit-learn-contrib/categorical-encoding.git
cd categorical-encoding
pip install -e .
cd ..
```

Download the data:

```
python download_data.py
```

Brief overview of the datasets is given in [overview_datasets.ipynb](overview_datasets.ipynb).

Run the benchmarks:

```
python main_test.py
```

**TODO:**

* datasets: look for other appropriate datasets. Current ideas:
    * add Criteo Terabyt Click Log dataset
    * generated dataset (both with uniform distribution as one with rare catgories)

  And expand overview of the categories in each dataset.

* Add the LeaveOneOutEncoder from category_encoders, and the CountFeaturizer
  from the sklearn PR.

* Investigate the different options:

  * Check the different implementations and what the differences are.

  * More clearly benchmark the different options (with/without shrinking,
    with/without cross-validation, different hyperparameters, ..), and investigate
    the different results for those.

Overview of initial runs of the benchmark are in [overview_results.ipynb](overview_results.ipynb) ([on nbviewer](http://nbviewer.jupyter.org/github/jorisvandenbossche/target-encoder-benchmarks/blob/master/overview_results.ipynb))(but the results still need to be investigated).


Benchmark code based on the provided code by Patricio Cerda et al (2018): https://arxiv.org/pdf/1806.00979.pdf (*"Similarity encoding for learning with dirty categorical variables"*).
