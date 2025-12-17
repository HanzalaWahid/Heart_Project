from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
import joblib

def train_logistic_regression(X_train , Y_train):
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    grid = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid.best_estimator_

def train_random_forest(X_train , Y_train):
    from sklearn.model_selection import RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)
    rand.fit(X_train, Y_train)
    return rand.best_estimator_

def train_svm(X_train , Y_train):
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid.best_estimator_

 
def save_model(model , file_path):
    joblib.dump(model , file_path)

