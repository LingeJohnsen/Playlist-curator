import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import average_precision_score, make_scorer

import matplotlib.pyplot as plt

#%%
# Functions

# Learning curve plot from sklearn documentation page
def plot_learning_curve(
                        estimator,
                        title, 
                        X, 
                        y, 
                        axes=None,
                        ylim=None,
                        cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)
                       ):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

#%% 
# Prepare dataset

df = pd.read_csv('jazz.csv', sep='|')

df = df.sample(frac=1, random_state=123).reset_index(drop=True)

#%%
num_cols = [
            'danceability',
            'energy',
            'speechiness',
            'acousticness',
            'instrumentalness',
            'liveness',
            'valence',
            'beats_duration_mean',
            'beats_duration_var',
            'bars_duration_var',
            'mode_var',
            'segments_duration_mean',
            'pitches_mean',
            'pitches_var',
            'tatums_duration_mean',
            'tatums_duration_var',
            'num_samples',
            'loudness',
            'tempo',
            'sections_num',
            'sections_duration_mean',
            'sections_duration_var',
            'loudness_var',
            'tempo_var',
            'key_var',
            'segments_num',
            'timbre_mean',
            'timbre_var',
            'tatums_num',
            'bars_num',
            'bars_duration_mean',
            'beats_num'  
           ]

cat_cols = [
            'time_signature',
            'key',
            'mode'
            ]
df = df.dropna(subset = cat_cols)

features = num_cols + cat_cols

X = df[features].copy()
y = df['label'].copy()

#%%
# Prepare training pipeline

NUM_TRIALS = 10

num_raw = [
           'danceability',
           'energy',
           'speechiness',
           'acousticness',
           'instrumentalness',
           'liveness',
           'valence',
           'beats_duration_mean',
           'beats_duration_var',
           'bars_duration_var',
           'mode_var',
           'segments_duration_mean',
           'pitches_mean',
           'pitches_var',
           'tatums_duration_mean',
           'tatums_duration_var'
          ]

num_scale = [
             'num_samples',
             'loudness',
             'tempo',
             'sections_num',
             'sections_duration_mean',
             'sections_duration_var',
             'loudness_var',
             'tempo_var',
             'key_var',
             'segments_num',
             'timbre_mean',
             'timbre_var',
             'tatums_num',
             'bars_num',
             'bars_duration_mean',
             'beats_num'          
            ]

# Set up pipeline
minmax_scaler = Pipeline(
                         steps = [
                                  ('imputer', SimpleImputer(strategy='median')),
                                  ('scaler', MinMaxScaler())
                                 ]
                        )
encoder = Pipeline(
                   steps = [
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encoder', OrdinalEncoder())
                           ]
                  )

raw_imputer = Pipeline(steps = [('imputer', SimpleImputer(strategy='median'))])

preprocessor = ColumnTransformer(
                                 transformers = [
                                                 ('enc', encoder, cat_cols),
                                                 ('std', minmax_scaler, num_scale),
                                                 ('imp', raw_imputer, num_raw)
                                                ],
                                 remainder = 'drop'
                                )

pipe = Pipeline(
                steps=[
                       ('preprocessor', preprocessor),
                       ('classifier', LogisticRegression(
                                                         penalty = 'elasticnet',
                                                         class_weight = 'balanced',
                                                         random_state = 42,
                                                         solver = 'saga',
                                                         max_iter = 4000,
                                                         l1_ratio = 0.8
                                                        ))
                      ]
               )
#%%

# Small dataset size so set up nested cross-validation
inner_k = 3
outer_k = 5
nested_scores = []
param_grid = {
              'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
             }
for i in range(NUM_TRIALS):
    
    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=i)
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=i+100)
    
    clf = GridSearchCV(
                       estimator = pipe,
                       param_grid = param_grid,
                       cv = inner_cv,
                       scoring = make_scorer(average_precision_score),
                       n_jobs = 4
                      )
    
    nested_score = cross_val_score(
                                   clf, 
                                   X = X,
                                   y = y,
                                   cv = outer_cv,
                                   verbose = 2,
                                   n_jobs = 4
                                  )
    nested_scores += list(nested_score)
    

model = clf.fit(X, y)
#%%
# Visualize model results

# Plot loss values

# Plot training and test scores

# Plot roc and precision recall

# Learning curve
# fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# title = 'Learning curve (Logistic regression)'
# learning_curve_fig = plot_learning_curve(
#                                          model,
#                                          title,
#                                          X,
#                                          y,
#                                          cv = inner_cv,
#                                          n_jobs = 4
#                                         )

#%% Get some model metrics

# Recall
# Precision
# Area under PR
# Area under ROC


#%%
# train_sizes, train_scores, test_scores, fit_times, _ = \
#         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
#                        train_sizes=train_sizes,
#                        return_times=True)