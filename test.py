# Model
from models.ai_models.landmark_detection_models.aim001_holistic_detection import Holistic_Detection_Model
from models.ai_models.gesture_recognition_models.aim006_gesture_recognition import Gesture_Recognition_Model

from models.calculation_models.cal001_face import FaceCalculationModel

from models.camera_models.cam001_imageRead import ImageReadModel
from models.camera_models.cam002_VideoRead import VideoReadModel

from models.pipeline_models.pip001_client import PipeLineClientModel

import cv2
import json
import struct
import signal
import time
import datetime

import numpy as np

# Mediapipe
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

terminated = False

video_path = "test.mp4"

# Define the font parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_color = (0, 0, 255)  # White color
thickness = 1

fps_coord = (50,50)
lefthandgestureText_coord = (50,100)

def signal_handler(sig, frame):
    global terminated
    print('Terminate Program!')
    camera_model.stopThread()
    terminated = True
    exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # Setup PipeLine connection
    # pipeLineModel = PipeLineClientModel(pipe_address=r'\\.\pipe\PythonCSharpPipe')
    # pipeLineModel.setUpPipeLineConnection()

    # Set up AI models
    holistic_model = Holistic_Detection_Model()
    gesture_model = Gesture_Recognition_Model()

    # Set up Calculation Model
    faceCal_model = FaceCalculationModel()

    # Set up Camera Stream Model
    # camera_model = ImageReadModel()
    camera_model = VideoReadModel(video_path)
    camera_model.startThread()

    height = 0
    width = 0
    channels = 0
    while not terminated:
        try:
            image = camera_model.getImage()
            height, width, channels = image.shape
            print("get first Image")
        except:
            print("No frame can be read!")
            continue
        break

    holistic_model.setIMGshape(width=width, height=height)
    faceCal_model.setIMGshape(width=width, height=height)
    last_time = time.time()
    counter = 0
    packet_error = 0
    while not terminated:
        try:
            image = camera_model.getImage()

            # TODO: Multiple thread running AI models?
            holistic_model.detect(image)
            
            # Pose landmark init
            pose_landmarks = holistic_model.raw_pose_landmarks
            # TODO: Implement Body tilting methods

            # Facelandmark init
            face_point = np.array(holistic_model.face_landmarks).astype('float32')
            face_3D_point = np.array(holistic_model.face_world_landmarks).astype('float32')
            faceCal_model.calculateVectors(face_point=face_point, face_3D_point=face_3D_point)
            faceCal_model.calMouthAspectRatio(face_point)
            faceCal_model.calEyesAspectRatio(face_point)

            # TODO: Implement Gesture recognition detect
            # Obtain Gesture ID
            gesture_model.lefthand_gesture_detect(holistic_model.lefthand_landmarks)
            lefthand_gestureID = gesture_model.lefthand_gestureID

            packet = {
                "landmark": holistic_model.raw_pose_landmarks,
                "HeadRollPitchYaw": faceCal_model.RollPitchYaw,
                "mar": faceCal_model.mar,
                "r_ear": faceCal_model.right_ear,
                "l_ear": faceCal_model.left_ear,
                "actionID": lefthand_gestureID,
                "id": counter,
            }
            packet = json.dumps(packet)
            
            # if (not (pipeLineModel.send_Packet_To_Unity(packet=packet))):
            #     packet_error += 1
            #     if (packet_error >= 20):
            #         camera_model.stopThread()
            #         break

            # (Debug use) Code for drawing landmarks on image input
            if (holistic_model.results.pose_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    holistic_model.results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            if (holistic_model.results.pose_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    holistic_model.results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                    
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=holistic_model.results.face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=holistic_model.results.face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time

            image = cv2.flip(image, 1)
            cv2.putText(image, str(fps), fps_coord, font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.putText(image, gesture_model.lefthand_gesture, lefthandgestureText_coord, font, font_scale, font_color, thickness, cv2.LINE_AA)

            cv2.imshow('MediaPipe Pose', image)
            counter += 1

            if cv2.waitKey(2) & 0xFF == 27:
                camera_model.stopThread()
                break
        except Exception as e:
            print(e)