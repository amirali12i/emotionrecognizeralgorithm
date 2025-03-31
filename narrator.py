import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from docx import Document
import matplotlib.pyplot as plt
import time
from collections import Counter
import logging

class WebcamAnalyzer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open webcam!")
            exit()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)
        self.pose = self.mp_pose.Pose()
        self.yolo_model = YOLO("yolov8n.pt")
        self.movement_history = []
        self.mood_history = []
        self.activity_history = []
        self.confidence_history = []
        self.posture_history = []
        self.expression_history = []
        self.head_orientation_history = []
        self.frame_counter = 0
        self.last_head_position = None
        self.current_activity = "unknown"
        self.activity_confidence = 0.0
        self.eye_state = "unknown"
        self.has_glasses = False
        self.room_lighting = "unknown"
        self.visual_attention = "center"
        self.background_objects = []
        self.posture = "unknown"
        self.expression = "unknown"
        self.head_orientation = "unknown"
        # Setup logging
        logging.basicConfig(filename='webcam_analysis.log', level=logging.INFO,
                            format='%(asctime)s - %(message)s')
        logging.info("WebcamAnalyzer initialized")

    def detect_attributes(self, frame, landmarks):
        h, w, _ = frame.shape
        eye_area = frame[int(landmarks.landmark[33].y * h) - 20:int(landmarks.landmark[263].y * h) + 20,
                         int(landmarks.landmark[33].x * w) - 20:int(landmarks.landmark[263].x * w) + 20]
        if eye_area.size > 0:
            gray = cv2.cvtColor(eye_area, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.has_glasses = np.sum(edges) / edges.size > 0.01
        left_eye = abs(landmarks.landmark[159].y - landmarks.landmark[145].y)
        right_eye = abs(landmarks.landmark[386].y - landmarks.landmark[374].y)
        avg = (left_eye + right_eye) / 2
        self.eye_state = "closed" if avg < 0.02 else "open"

    def detect_head_movement(self, landmarks):
        nose = (landmarks.landmark[1].x, landmarks.landmark[1].y)
        if self.last_head_position:
            move = np.sqrt((nose[0] - self.last_head_position[0])**2 + (nose[1] - self.last_head_position[1])**2)
            self.movement_history.append(move)
        self.last_head_position = nose

    def analyze_movement_intensity(self):
        if len(self.movement_history) > 0:
            avg_movement = np.mean(self.movement_history[-10:])
            if avg_movement < 0.01:
                return "low"
            elif avg_movement < 0.05:
                return "medium"
            else:
                return "high"
        return "unknown"

    def detect_objects(self, frame):
        results = self.yolo_model(frame)
        objects = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                if box.conf > 0.5 and self.yolo_model.names[int(box.cls)] != "person":
                    obj_name = self.yolo_model.names[int(box.cls)].lower()
                    objects.append(obj_name)
                    logging.info(f"Object detected: {obj_name}")
        self.background_objects = objects
        return objects

    def detect_activity(self, pose_landmarks):
        left_wrist = pose_landmarks.landmark[15].y
        right_wrist = pose_landmarks.landmark[16].y
        nose = pose_landmarks.landmark[0].y
        if "phone" in self.background_objects and (abs(left_wrist - nose) < 0.1 or abs(right_wrist - nose) < 0.1):
            self.current_activity = "using a phone"
            self.activity_confidence = 0.9
        elif "book" in self.background_objects and left_wrist < nose and right_wrist < nose:
            self.current_activity = "reading a book"
            self.activity_confidence = 0.8
        else:
            self.current_activity = "sitting"
            self.activity_confidence = 0.7
        self.activity_history.append(self.current_activity)
        self.confidence_history.append(self.activity_confidence)

    def analyze_posture(self, pose_landmarks):
        if pose_landmarks:
            left_shoulder = pose_landmarks.landmark[11]
            right_shoulder = pose_landmarks.landmark[12]
            nose = pose_landmarks.landmark[0]
            shoulder_avg_z = (left_shoulder.z + right_shoulder.z) / 2
            self.posture = "leaning forward" if nose.z < shoulder_avg_z - 0.05 else "straight"
        else:
            self.posture = "unknown"
        self.posture_history.append(self.posture)

    def analyze_expression(self, face_landmarks):
        if face_landmarks:
            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]
            mouth_center = face_landmarks.landmark[13]
            avg_y = (left_mouth.y + right_mouth.y) / 2
            if avg_y < mouth_center.y - 0.01:
                self.expression = "smile"
            elif avg_y > mouth_center.y + 0.01:
                self.expression = "frown"
            else:
                self.expression = "neutral"
        else:
            self.expression = "unknown"
        self.expression_history.append(self.expression)

    def detect_head_orientation(self, face_landmarks):
        if face_landmarks:
            nose_x = face_landmarks.landmark[1].x
            left_ear_x = face_landmarks.landmark[234].x
            right_ear_x = face_landmarks.landmark[454].x
            if nose_x < (left_ear_x + right_ear_x) / 2 - 0.1:
                self.head_orientation = "left"
            elif nose_x > (left_ear_x + right_ear_x) / 2 + 0.1:
                self.head_orientation = "right"
            else:
                self.head_orientation = "forward"
        else:
            self.head_orientation = "unknown"
        self.head_orientation_history.append(self.head_orientation)

    def estimate_mood(self):
        score = 0
        if self.eye_state == "open":
            score += 1
        if len(self.movement_history) > 0:
            avg_movement = np.mean(self.movement_history[-10:])
            if avg_movement < 0.01:
                score += 1
            elif avg_movement > 0.05:
                score -= 1
        if self.activity_confidence > 0.8:
            score += 1
        else:
            score -= 1
        if self.posture == "straight":
            score += 1
        if self.expression == "smile":
            score += 1
        elif self.expression == "frown":
            score -= 1
        if score >= 3:
            mood = "focused"
        elif score >= 1:
            mood = "neutral"
        else:
            mood = "distracted"
        logging.info(f"Mood estimated: {mood} (score: {score})")
        return mood

    def estimate_attention(self, landmarks):
        gaze_x = (landmarks.landmark[33].x + landmarks.landmark[263].x) / 2
        if gaze_x < 0.4:
            return "left"
        elif gaze_x > 0.6:
            return "right"
        else:
            return "center"

    def detect_room_lighting(self, frame):
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if brightness > 150:
            self.room_lighting = "bright"
        elif brightness > 100:
            self.room_lighting = "moderate"
        else:
            self.room_lighting = "dim"

    def generate_narrative(self):
        text = f"The student was in a {self.room_lighting}-lit room. "
        if self.has_glasses:
            text += "They had glasses on. "
        text += f"Their eyes were mostly {self.eye_state}. "
        if self.activity_history:
            predominant_activity = Counter(self.activity_history).most_common(1)[0][0]
            text += f"They were mostly {predominant_activity}. "
        if self.mood_history:
            mood_counter = Counter(self.mood_history)
            total = len(self.mood_history)
            focused_percent = (mood_counter["focused"] / total) * 100 if "focused" in mood_counter else 0
            neutral_percent = (mood_counter["neutral"] / total) * 100 if "neutral" in mood_counter else 0
            distracted_percent = (mood_counter["distracted"] / total) * 100 if "distracted" in mood_counter else 0
            text += f"They were focused in {focused_percent:.1f}% of the frames, neutral in {neutral_percent:.1f}%, and distracted in {distracted_percent:.1f}%. "
        if self.posture_history:
            valid_postures = [p for p in self.posture_history if p != "unknown"]
            if valid_postures:
                predominant_posture = Counter(valid_postures).most_common(1)[0][0]
                text += f"They mostly sat {predominant_posture}. "
        if self.expression_history:
            valid_expressions = [e for e in self.expression_history if e != "unknown"]
            if valid_expressions:
                predominant_expression = Counter(valid_expressions).most_common(1)[0][0]
                text += f"They had a {predominant_expression} expression most of the time. "
        if self.head_orientation_history:
            valid_orientations = [o for o in self.head_orientation_history if o != "unknown"]
            if valid_orientations:
                predominant_orientation = Counter(valid_orientations).most_common(1)[0][0]
                text += f"Their head was mostly facing {predominant_orientation}. "
        if self.movement_history:
            movement_intensity = self.analyze_movement_intensity()
            text += f"Their movement intensity was {movement_intensity}. "
        if self.background_objects:
            objects = ", ".join(set(self.background_objects))
            text += f"Objects such as {objects} were detected."
        print("Narrative:", text)
        doc = Document()
        doc.add_paragraph(text)
        doc.save("narrative.docx")

    def create_visualizations(self):
        plt.figure(figsize=(12, 15))
        plt.subplot(5, 1, 1)
        moods = [1 if m == "distracted" else 2 if m == "neutral" else 3 for m in self.mood_history]
        plt.plot(moods)
        plt.title("Mood Over Time")
        plt.subplot(5, 1, 2)
        if self.movement_history:
            plt.hist(self.movement_history, bins=20)
            plt.title("Movement Distribution")
        plt.subplot(5, 1, 3)
        plt.plot(self.confidence_history)
        plt.title("Activity Confidence")
        plt.subplot(5, 1, 4)
        postures = [1 if p == "straight" else 0 if p == "leaning forward" else -1 for p in self.posture_history]
        plt.plot(postures)
        plt.title("Posture Over Time")
        plt.subplot(5, 1, 5)
        expressions = [1 if e == "smile" else 0 if e == "neutral" else -1 if e == "frown" else -2 for e in self.expression_history]
        plt.plot(expressions)
        plt.title("Expression Over Time")
        plt.tight_layout()
        plt.savefig("visualizations.png")
        plt.close()

    def run(self):
        start = time.time()
        while time.time() - start < 25:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame grab failed!")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)
            face = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
            pose = pose_results.pose_landmarks if pose_results else None
            if face:
                self.detect_attributes(frame, face)
                self.analyze_expression(face)
                self.detect_head_movement(face)
                self.detect_head_orientation(face)
                self.visual_attention = self.estimate_attention(face)
            else:
                self.expression = "unknown"
                self.head_orientation = "unknown"
                self.expression_history.append(self.expression)
                self.head_orientation_history.append(self.head_orientation)
            if pose:
                self.analyze_posture(pose)
                self.detect_activity(pose)
            else:
                self.posture = "unknown"
                self.posture_history.append(self.posture)
            mood = self.estimate_mood()
            self.mood_history.append(mood)
            self.detect_room_lighting(frame)
            self.detect_objects(frame)
            self.frame_counter += 1
            # Display info on frame
            cv2.putText(frame, f"Frame: {self.frame_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mood: {mood}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Posture: {self.posture}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Expression: {self.expression}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        self.generate_narrative()
        self.create_visualizations()

if __name__ == "__main__":
    analyzer = WebcamAnalyzer()
    analyzer.run()