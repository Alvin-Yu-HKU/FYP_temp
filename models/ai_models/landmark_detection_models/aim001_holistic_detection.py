import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

class Holistic_Detection_Model:

    def __init__ (self):
        self.model_hold = False
        # Set up mediapipe holistic model
        self.holistic_model = mp_holistic.Holistic(
            model_complexity=2,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5)

        self.imgWidth = -1
        self.imgHeight = -1

    def setIMGshape(self, width, height):
        self.imgWidth = width
        self.imgHeight = height

    # Function for doing holistic landmark detection and retreive results
    def detect(self, image):
        if (self.model_hold):
            return

        self.model_hold = True
        self.results = self.holistic_model.process(image)
        # Update Pose Landmarks results
        if self.results.pose_landmarks:
            self.pose_landmarks = []
            self.pose_world_landmarks = []
            self.raw_pose_landmarks = []
            self.pose_landmarks_visibility = []
            
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                x, y = int(landmark.x * self.imgWidth), int(landmark.y * self.imgHeight)
                self.pose_landmarks.append([x, y])
                self.raw_pose_landmarks.append([landmark.x, 1 - landmark.y, landmark.z*0.1])
                self.pose_world_landmarks.append([x, y, landmark.z])
                self.pose_landmarks_visibility.append(landmark.visibility)
            
            # Debug use
            # print(f"Pose: {self.pose_landmarks[0]}")

        # Update Face Landmarks and Iris Landmarks results
        self.face_landmarks = []
        self.raw_face_landmarks = []
        self.face_world_landmarks = []
        if self.results.face_landmarks:
            for landmark in self.results.face_landmarks.landmark[:468]:
                x, y = int(landmark.x * self.imgWidth), int(landmark.y * self.imgHeight)
                self.face_landmarks.append([x, y])
                self.raw_face_landmarks.append([landmark.x, landmark.y, landmark.z])
                self.face_world_landmarks.append([x, y, landmark.z])

        # TODO: Testing on iris calculation
        #     self.iris_landmarks = []
        #     self.iris_world_landmarks = []
        #     for landmark in self.results.face_landmarks.landmark[-10:]:
        #         x, y = int(landmark.x * self.imgWidth), int(landmark.y * self.imgHeight)
        #         self.iris_landmarks.append([x, y])
        #         self.iris_world_landmarks.append([x, y, landmark.z])

        #     # Debug use
        #     # print(f"Face: {self.face_landmarks}")
        #     # print(f"Iris: {self.iris_landmarks}")

        # Update Hand Landmarks results
        self.lefthand_landmarks = []
        self.raw_lefthand_landmarks = []
        self.lefthand_world_landmarks = []
        if self.results.left_hand_landmarks:
            for landmark in self.results.left_hand_landmarks.landmark:
                x, y = int(landmark.x * self.imgWidth), int(landmark.y * self.imgHeight)
                self.lefthand_landmarks.append([x ,y])
                self.raw_lefthand_landmarks.append([landmark.x, landmark.y])
                self.lefthand_world_landmarks.append([x, y, landmark.z])

            # Debug use
            # print(f"Left Hand: {self.left_hand_landmarks[0]}")
        self.right_hand_landmarks = []
        self.raw_right_hand_landmarks = []
        self.right_hand_world_landmarks = []
        if self.results.right_hand_landmarks:
            for landmark in self.results.right_hand_landmarks.landmark:
                x, y = int(landmark.x * self.imgWidth), int(landmark.y * self.imgHeight)
                self.right_hand_landmarks.append([x, y])
                self.raw_right_hand_landmarks.append([landmark.x, landmark.y])
                self.right_hand_world_landmarks.append([x, y, landmark.z])
                
        #     # Debug use
        #     # print(f"Right Hand: {self.right_hand_landmarks[0]}")

        self.model_hold = False
