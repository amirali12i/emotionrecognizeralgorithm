import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        self.landmark_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 130, 243, 112, 23, 27, 30, 247, 375, 321, 405, 314, 17, 181, 91, 61, 291, 0, 267, 269, 270, 409]  # 30 key landmarks

    def extract_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for idx, lm in enumerate(results.multi_face_landmarks[0].landmark) if idx in self.landmark_indices], dtype=np.float32)
            return landmarks.flatten() if len(landmarks) == 30 else None
        return None

    def process_dataset(self):
        for split in ["train", "test"]:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                print(f"Error: {split} directory not found in {self.data_dir}")
                continue

            for emotion_dir in split_dir.iterdir():
                if emotion_dir.is_dir():
                    emotion = emotion_dir.name
                    processed_emotion_dir = self.processed_dir / split / emotion
                    processed_emotion_dir.mkdir(parents=True, exist_ok=True)

                    for image_path in emotion_dir.glob("*.jpg"):
                        image = cv2.imread(str(image_path))
                        if image is None:
                            print(f"Failed to load {image_path}")
                            continue

                        landmarks = self.extract_landmarks(image)
                        if landmarks is not None:
                            np.save(processed_emotion_dir / f"{image_path.stem}_landmarks.npy", landmarks)
                            print(f"Processed {image_path} with landmarks saved to {processed_emotion_dir / f'{image_path.stem}_landmarks.npy'}")
                        else:
                            print(f"No landmarks detected for {image_path}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process_dataset()