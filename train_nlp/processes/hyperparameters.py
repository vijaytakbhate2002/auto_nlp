from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Hyperparameter combinations for RandomForestClassifier
random_forest_params = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [None, 10, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy", "log_loss"]
}

# Hyperparameter combinations for LogisticRegression
logistic_regression_params = {
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "saga", "lbfgs", "newton-cg"],
    "max_iter": [100, 200, 500],
    "class_weight": [None, "balanced"]
}

# Hyperparameter combinations for DecisionTreeClassifier
decision_tree_params = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"]
}

# Hyperparameter combinations for MultinomialNB
multinomial_nb_params = {
    "alpha": [0.01, 0.1, 1, 10],
    "fit_prior": [True, False]
}

models_with_params = {
    "rf": (RandomForestClassifier(), random_forest_params, 'random forest'),
    "lr": (LogisticRegression(), logistic_regression_params, 'logistic regression'),
    "dt": (DecisionTreeClassifier(), decision_tree_params, 'decision tree'),
    "nb": (MultinomialNB(), multinomial_nb_params, 'multinomial nb')
}