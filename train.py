from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
from pathlib import Path

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        self.models_dir = Path("models")
        self.processed_dir = Path("data/processed")
        self.models_dir.mkdir(exist_ok=True)

    def load_data(self):
        X, y = [], []
        emotion_map = {
            "happy": "joyful", "sad": "melancholy", "angry": "irritated",
            "neutral": "peaceful", "surprise": "excited", "fear": "anxious", "disgust": "confused"
        }

        for split in ["train", "test"]:
            split_dir = self.processed_dir / split
            if not split_dir.exists():
                print(f"Error: {split} directory not found in {self.processed_dir}")
                continue

            for emotion_dir in split_dir.iterdir():
                if emotion_dir.is_dir():
                    fer_emotion = emotion_dir.name
                    mapped_emotion = emotion_map.get(fer_emotion, fer_emotion)
                    for landmark_path in emotion_dir.glob("*_landmarks.npy"):
                        landmarks = np.load(landmark_path)
                        X.append(landmarks)
                        y.append(mapped_emotion)

        return np.array(X), np.array(y)

    def train(self):
        X, y = self.load_data()
        if len(X) == 0:
            print("No processed data found. Run data.py first.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Training completed. Accuracy on test set: {accuracy:.4f}, Samples: {len(X)}")
        joblib.dump(self.model, self.models_dir / "emotion_classifier.pkl")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()