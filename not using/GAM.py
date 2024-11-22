# Create a scikit-learn compatible GAM wrapper
class GAMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, lam=1.0, n_splines=10):
        self.lam = lam
        self.n_splines = n_splines
        self.model = None

    def fit(self, X, y):
        # Initialize GAM with the specified parameters
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        self.model = LinearGAM(s(0, n_splines=self.n_splines, lam=self.lam))
        self.model.gridsearch(X, y)  # Automatically tunes hyperparameters
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        # Use negative mean squared error as the score
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)
    
    
#---------------- 2.GAM---------------------------

gam_model = GAMWrapper()
param_grid_gam = {
    'lam': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'n_splines': [5, 10, 20, 50],    # Number of splines for each feature
}
# Perform randomized search for hyperparameter tuning
random_search_gam = RandomizedSearchCV(
    estimator=gam_model,
    param_distributions=param_grid_gam,
    n_iter=10,  # Number of parameter settings sampled
    scoring='neg_mean_squared_error',
    cv=5,  # 5-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1,
    error_score='raise'  # Raise errors during fit

)

# Train
nestedCV(random_search_gam, "GAM")

