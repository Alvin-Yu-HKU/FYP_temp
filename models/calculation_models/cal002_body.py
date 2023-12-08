import cv2
import math
import numpy as np

class BodyCalculationModel:
    def __init__(self):
        self.bodylandmark = [11, 12, 23, 24]
        self.footlandmark = [25,26,27,28,29,30,31]

        # TODO: Further testing on tilting of body
        self.foot_tilt_angle = 35
        self.body_tilt_angle = 10

    def coord_tilt(self, landmark, angle):
        angle_radians = np.radians(angle)

        # Define the rotation matrix along x axis
        rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                [0, np.sin(angle_radians), np.cos(angle_radians)]])

        rotated_coordinate = np.dot(rotation_matrix, np.array(landmark))
        
        return rotated_coordinate.tolist()

    def tilt_body(self, landmarks):
        for idx in self.bodylandmark:
            rotated_coordinates = self.coord_tilt(landmark=landmarks[idx], angle=self.body_tilt_angle)
            landmarks[idx] = rotated_coordinates
        return landmarks

    def tilt_foot(self, landmarks):
        for idx in self.footlandmark:
            rotated_coordinates = self.coord_tilt(landmark=landmarks[idx], angle=self.foot_tilt_angle)
            landmarks[idx] = rotated_coordinates
        return landmarks

    def tilt_wholeBody(self, landmarks):
        for idx, landmark in enumerate(landmarks):
            rotated_coordinates = self.coord_tilt(landmark=landmark, angle=20)
            landmarks[idx] = rotated_coordinates
        return landmarks

    def calculateBodyAimTarget(self, landmarks):
        landmarks = self.tilt_body(landmarks=landmarks)

        left_shoulder = np.array(landmarks[11])
        right_shoulder = np.array(landmarks[12])
        body_top =  (left_shoulder + right_shoulder) / 2

        left_hip = np.array(landmarks[23])
        right_hip = np.array(landmarks[24])
        body_bottom = (left_hip + right_hip) / 2

        body_aimTarget = (body_top - body_bottom).tolist()
        return body_aimTarget

