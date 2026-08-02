import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class PoseClassifier:
    def __init__(self):
        self.mlp = MLPClassifier()
        self.scaler = StandardScaler()
        self._trained = False
    def train(self, features, labels):
        scaled_features = self.scaler.fit_transform(features)
        self.mlp.fit(scaled_features, labels)
        self._trained = True
    def predict(self, features):
        features = np.asarray(features)
        if not self._trained:
            raise RuntimeError("No model trained yet.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = self.scaler.transform(features)
        return self.mlp.predict(features)
    def predict_proba(self, features):
        features = np.asarray(features)
        if not self._trained:
            raise RuntimeError("No model trained yet.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = self.scaler.transform(features)
        return self.mlp.predict_proba(features)
    def save(self, path):
        joblib.dump({'mlp': self.mlp, 'scaler': self.scaler, '_trained': self._trained}, path)
    @classmethod
    def load(cls, path):
        joblib_data = joblib.load(path)
        classifier = cls()
        classifier.mlp = joblib_data['mlp']
        classifier.scaler = joblib_data['scaler']
        classifier._trained = joblib_data['_trained']
        return classifier
