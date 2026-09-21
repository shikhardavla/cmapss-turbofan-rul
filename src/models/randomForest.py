from sklearn.ensemble import RandomForestRegressor
from src.models.base import BaseModel
import joblib

class RandomForestModel (BaseModel):

    """
    A BaseModel compatible wrapper for the Random Forest module
    """

    def __init__(self, **params):
        self.model = RandomForestRegressor (**params)
        self.feature_names = None
    
    def train (self, X_train, y_train, **kwargs):

        # Fits RFRegressor and records feature names (sensor statistics per engine window, and cluster ID learned from k-means clustering)

        if hasattr (X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        self.model.fit(X_train, y_train)

    def predict (self, X):

        # Predicts RUL on test data. Re-orders features to align with train set.

        if self.feature_names is not None and hasattr(X, 'columns'):
            X = X[self.feature_names]
            
        return self.model.predict(X)
    def save(self, path):
        # Saves the RF model artifact for evaluation and deployment
     
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names
        }, path)

    def load(self, path):
        # Retrieves the RF model artifact for deployment/production
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        return self 
        



