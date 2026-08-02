import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from classifier import PoseClassifier

def train_model(csv_path, model_path):
    csv_path = Path(csv_path)
    model_path = Path(model_path)
    df = pd.read_csv(csv_path)
    templates = {}
    for label in df["label"].unique():
        pose_df = df[df["label"] == label]
        mean_angles = pose_df.drop(columns=["label"]).mean().tolist()
        templates[label] = mean_angles
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = PoseClassifier()
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    classifier.save(model_path)
    templates_path = model_path.parent / "pose_templates.joblib"
    joblib.dump(templates, templates_path)
    print(f"Templates saved to {templates_path}")
    print(f"Model saved to {model_path}")
    return classifier

if __name__ == "__main__":
    train_model("data/features.csv", "models/pose_classifier.joblib")