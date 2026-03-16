# models/optimization.py
from sklearn.model_selection import GridSearchCV

def optimize_logistic(pipeline, X_train, y_train):
    """Optimize Logistic Regression hyperparameters"""
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
       # 'classifier__solver': ['newton-cg', 'lbfgs', 'sag'],
        'classifier__max_iter': [500, 1000]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best Logistic params: {grid_search.best_params_}")
    print(f"Best Logistic CV Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def optimize_random_forest(pipeline, X_train, y_train):
    """Optimize Random Forest hyperparameters"""
    param_grid = {
        'classifier__n_estimators': [25, 50],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best Random Forest params: {grid_search.best_params_}")
    print(f"Best Random Forest CV Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def optimize_xgboost(pipeline, X_train, y_train):
    """Optimize XGBoost hyperparameters"""
    param_grid = {
        'classifier__n_estimators': [25, 50],
        'classifier__max_depth': [3, 6],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best XGBoost params: {grid_search.best_params_}")
    print(f"Best XGBoost CV Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_