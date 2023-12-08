import cv2
import math
import numpy as np

class FaceCalculationModel:

    def __init__(self):
        self.eye_aspect_ratio = -1

        self.rightEyelandmark = {
            "upper": [246, 161, 160, 159, 158, 157, 173],
            "lower": [33, 7, 163, 144, 145, 153, 154, 155, 133]
            }
        self.leftEyelandmark = {
            "upper": [466, 388, 387, 386, 385, 384, 398],
            "lower": [263, 249, 390, 373, 374, 380, 381, 382, 362]
        }
        self.mouthlandmark = {
            "upper": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
            "lower": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        }

        self.right_ear = -1
        self.left_ear = -1

        self.ear_max = 0.35
        self.ear_min = 0.1

        self.mar = -1
        self.mouth_distance = -1

        self.iris_x_scale_factor = 2.2
        self.iris_y_scale_factor = 3.5

        self.r_iris_x_rate = -1
        self.r_iris_y_rate = -1

        self.l_iris_x_rate = -1
        self.l_iris_y_rate = -1

        # Assuming no lens distortion
        self.dist_matrix = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.rotation_vector = None
        self.translation_vector = None

    def setIMGshape(self, width, height):
        self.imgWidth = width
        self.imgHeight = height

        # Set up CameraMatrix when got image shape and height
        self.setCameraMatrix()

    def setCameraMatrix(self):
        # Camera internals
        self.focal_length = self.imgWidth
        self.camera_center = (self.imgWidth / 2, self.imgHeight / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[1]],
                [0, self.focal_length, self.camera_center[0]],
                [0, 0, 1]], dtype="double")

    def getDistance2D(self, x1, y1, x2, y2):
        # Calculate Euclidean Distance
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def map_ear_value(self, value):
        normalized_value = (value - self.ear_min) / (self.ear_max - self.ear_min)
        return normalized_value

    def calEyesAspectRatio(self, face_landmark):
        # Calculate Right EAR
        p1 = face_landmark[self.rightEyelandmark["lower"][0]]
        p2 = face_landmark[self.rightEyelandmark["upper"][2]]
        p3 = face_landmark[self.rightEyelandmark["upper"][4]]
        p4 = face_landmark[self.rightEyelandmark["lower"][8]]
        p5 = face_landmark[self.rightEyelandmark["lower"][5]]
        p6 = face_landmark[self.rightEyelandmark["lower"][3]]

        p2_To_p6 = self.getDistance2D(p2[0], p2[1], p6[0], p6[1])
        p3_To_p5 = self.getDistance2D(p3[0], p3[1], p5[0], p5[1])
        p1_To_p4 = self.getDistance2D(p1[0], p1[1], p4[0], p4[1])

        self.right_ear = self.map_ear_value((p2_To_p6 + p3_To_p5) / (2 * p1_To_p4))
    
        # Calculate Left EAR
        p1 = face_landmark[self.leftEyelandmark["lower"][8]]
        p2 = face_landmark[self.leftEyelandmark["upper"][4]]
        p3 = face_landmark[self.leftEyelandmark["upper"][2]]
        p4 = face_landmark[self.leftEyelandmark["lower"][0]]
        p5 = face_landmark[self.leftEyelandmark["lower"][3]]
        p6 = face_landmark[self.leftEyelandmark["lower"][5]]

        p2_To_p6 = self.getDistance2D(p2[0], p2[1], p6[0], p6[1])
        p3_To_p5 = self.getDistance2D(p3[0], p3[1], p5[0], p5[1])
        p1_To_p4 = self.getDistance2D(p1[0], p1[1], p4[0], p4[1])

        self.left_ear = self.map_ear_value((p2_To_p6 + p3_To_p5) / (2 * p1_To_p4))
    
    def getRightEAR(self):
        return self.map_ear_value(value=self.right_ear)

    def getLeftEAR(self):
        return self.map_ear_value(value=self.left_ear)

    def calMouthAspectRatio(self, face_landmark):
        p1 =  (face_landmark[self.mouthlandmark["upper"][0]] + face_landmark[self.mouthlandmark["lower"][0]]) / 2
        p2 = (face_landmark[self.mouthlandmark["upper"][2]] + face_landmark[self.mouthlandmark["upper"][3]]) / 2
        p3 = face_landmark[self.mouthlandmark["upper"][5]]
        p4 = (face_landmark[self.mouthlandmark["upper"][7]] + face_landmark[self.mouthlandmark["upper"][8]]) / 2
        p5 = (face_landmark[self.mouthlandmark["upper"][10]] + face_landmark[self.mouthlandmark["lower"][10]]) / 2
        p6 = (face_landmark[self.mouthlandmark["lower"][2]] + face_landmark[self.mouthlandmark["lower"][3]]) / 2
        p7 = face_landmark[self.mouthlandmark["lower"][5]]
        p8 = (face_landmark[self.mouthlandmark["lower"][7]] + face_landmark[self.mouthlandmark["upper"][8]]) / 2

        p2_To_p8 = self.getDistance2D(p2[0], p2[1], p8[0], p8[1])
        p3_To_p7 = self.getDistance2D(p3[0], p3[1], p7[0], p7[1])
        p4_To_p6 = self.getDistance2D(p4[0], p4[1], p6[0], p6[1])
        p1_To_p5 = self.getDistance2D(p1[0], p1[1], p5[0], p5[1])

        self.mar = (p2_To_p8 + p3_To_p7 + p4_To_p6) / (2 * p1_To_p5)

    def getMAR(self):
        return self.mar
    
    def calMouthDistance(self, face_landmark):
        mouth_left = face_landmark[self.mouthlandmark["upper"][0]]
        mouth_right = face_landmark[self.mouthlandmark["upper"][10]]
        self.mouth_distance =  self.getDistance2D(mouth_left[0], mouth_left[1], mouth_right[0], mouth_right[1])

    def getMouthDistance(self):
        return self.mouth_distance

    def calIrisRatio(self, face_landmark, iris_landmark):
        # Calculate how iris of right eye shift toward left or right (x -> 0: left, 1: right) and toward top and bootom (y -> 0: top, 1: bottom)
        right_eye_left = face_landmark[self.rightEyelandmark["lower"][0]]
        right_eye_right = face_landmark[self.rightEyelandmark["lower"][8]]

        right_eye_high = face_landmark[self.rightEyelandmark["upper"][3]]
        right_eye_low = face_landmark[self.rightEyelandmark["lower"][4]]

        r_p_iris = iris_landmark[0]

        # Calculate how iris of right eye shift toward left or right (x -> 0: left, 1: right)
        iris_To_right_eye_left = np.linalg.norm([r_p_iris[0] - right_eye_left[0], r_p_iris[1] - right_eye_left[1]])
        right_eye_right_To_right_eye_left = np.linalg.norm([right_eye_right[0] - right_eye_left[0], right_eye_right[1] - right_eye_left[1]])

        self.r_iris_x_rate = 1 - (right_eye_right_To_right_eye_left - iris_To_right_eye_left) / right_eye_right_To_right_eye_left

        iris_To_right_eye_high = np.linalg.norm([r_p_iris[0] - right_eye_high[0], r_p_iris[1] - right_eye_high[1]])
        right_eye_low_To_right_eye_high = np.linalg.norm([right_eye_low[0] - right_eye_high[0], right_eye_low[1] - right_eye_high[1]])

        if (right_eye_low_To_right_eye_high  > iris_To_right_eye_high):
            self.r_iris_y_rate = (right_eye_low_To_right_eye_high - iris_To_right_eye_high) / right_eye_low_To_right_eye_high
        else:
            self.r_iris_y_rate = 0
        # Calculate how iris of right eye shift toward top and bootom (y -> 0: top, 1: bottom)
        left_eye_left = face_landmark[self.leftEyelandmark["lower"][0]]
        left_eye_right = face_landmark[self.leftEyelandmark["lower"][8]]

        left_eye_high = face_landmark[self.leftEyelandmark["upper"][3]]
        left_eye_low = face_landmark[self.leftEyelandmark["lower"][4]]

        l_p_iris = iris_landmark[5]

        # Calculate how iris of left eye shift toward left or right (x -> 0: left, 1: right)
        iris_To_left_eye_left = np.linalg.norm([l_p_iris[0] - left_eye_left[0], l_p_iris[1] - left_eye_left[1]])
        left_eye_right_To_left_eye_left = np.linalg.norm([left_eye_right[0] - left_eye_left[0], left_eye_right[1] - left_eye_left[1]])

        self.l_iris_x_rate = (left_eye_right_To_left_eye_left - iris_To_left_eye_left) / left_eye_right_To_left_eye_left

        iris_To_left_eye_high = np.linalg.norm([l_p_iris[0] - left_eye_high[0], l_p_iris[1] - left_eye_high[1]])
        left_eye_low_To_left_eye_high = np.linalg.norm([left_eye_low[0] - left_eye_high[0], left_eye_low[1] - left_eye_high[1]])

        if (left_eye_low_To_left_eye_high > iris_To_left_eye_high):
            self.l_iris_y_rate = (left_eye_low_To_left_eye_high - iris_To_left_eye_high) / left_eye_low_To_left_eye_high
        else:
            self.l_iris_y_rate = 0

        self.changeirisRatioMapping()

        self.l_iris_x_rate *= self.iris_x_scale_factor
        self.r_iris_x_rate *= self.iris_x_scale_factor

        self.l_iris_y_rate *= self.iris_y_scale_factor
        self.r_iris_y_rate *= self.iris_y_scale_factor
         
    def changeirisRatioMapping(self):
        self.l_iris_x_rate = (self.l_iris_x_rate - 0.5) * 2 
        self.l_iris_y_rate = (self.l_iris_y_rate - 0.5) * 2 

        self.r_iris_x_rate = (self.r_iris_x_rate - 0.5) * 2
        self.r_iris_y_rate = (self.r_iris_y_rate - 0.5) * 2 

    def getRightIrysRatio(self):
        return self.r_iris_x_rate, self.r_iris_y_rate

    def getLeftIrysRatio(self):
        return self.l_iris_x_rate, self.l_iris_y_rate

    def calAllFeature(self, face_landmark, iris_landmark):
        self.calEyesAspectRatio(face_landmark)
        self.calMouthAspectRatio(face_landmark)
        self.calMouthDistance(face_landmark)
        self.calIrisRatio(face_landmark, iris_landmark)

    def calculateVectors(self, face_point, face_3D_point):
        """
        Solve pose from all the 468 image points
        Return (rotation_vector, translation_vector) as pose.
        """
        success, self.rotation_vector, self.translation_vector = cv2.solvePnP(
        face_3D_point,
        face_point,
        self.camera_matrix,
        self.dist_matrix)

        if (not success):
            print("failed")
            return

        # print(f"rotation vector: {rotation_vector}")
        rot_matrix, _ = cv2.Rodrigues(self.rotation_vector)
        
        # Perform the RQ decomposition of the rotation matrix using cv2.RQDecomp3x3
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_matrix)

        # Extract the Euler angles (pitch, yaw, roll) from the RQ decomposition
        pitch, yaw, roll = angles

        pitch *= 1500
        yaw *= 1500
        roll *= 1500

        self.pitch = np.clip(pitch, -90, 90)
        self.yaw = np.clip(yaw, -90, 90)
        self.roll = np.clip(roll, -90, 90)

        self.RollPitchYaw = [roll, pitch, yaw]
        return self.RollPitchYaw
