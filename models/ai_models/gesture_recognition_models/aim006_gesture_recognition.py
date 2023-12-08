import numpy as np
import pickle
import threading

import torch

class Gesture_Recognition_Model:

    def __init__ (self):
        self.model_hold = False

        self.lefthand_detect = False
        self.righthand_detect = False

        self.lefthand_pt_path = "./models/ai_models/gesture_recognition_models/lefthand/aim002_left_hand_gesture_recognition.pt"
        self.righthand_pt_path = "./models/ai_models/gesture_recognition_models/righthand/aim004_right_hand_gesture_recognition.pt"

        self.lefthand_scaler_path = "./models/ai_models/gesture_recognition_models/lefthand/aim003_left_hand_scaler.pkl"
        self.righthand_scaler_path = "./models/ai_models/gesture_recognition_models/righthand/aim005_right_hand_scaler.pkl"

        self.mapping = ["call","dislike","fist","four","like","mute","ok","one","palm","peace_inverted","peace","rock","stop_inverted","stop","three","three2","two_up","two_up_inverted", "none"]

        self.left_hand_landmarks = []
        self.right_hand_landmarks = []

        self.lefthand_threshold = 30
        self.righthand_threshold = 30

        self.lefthand_gestureID = -1
        self.righthand_gestureID = -1
        self.lefthand_gesture = "Null"
        self.righthand_gesture = "Null"

        # Initialize Pytorch Model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using {self.device}')

        device_idx = 0
        # Select the GPU using the set_device() function
        torch.cuda.set_device(device_idx)
        print(torch.cuda.current_device())

        # Define the left hand gesture recognition model
        self.lefthand_model = torch.jit.load(self.lefthand_pt_path, map_location=self.device)
        self.lefthand_model.eval()

        # Define the right hand gesture recognition model
        self.righthand_model = torch.jit.load(self.righthand_pt_path, map_location=self.device)
        self.righthand_model.eval()

        with open(self.lefthand_scaler_path, "rb") as file:
            self.lefthand_scaler = pickle.load(file)

        with open(self.righthand_scaler_path, "rb") as file:
            self.righthand_scaler = pickle.load(file)

    def lefthand_gesture_detect(self, lefthand_landmarks):
        if (len(lefthand_landmarks) == 0):
            self.lefthand_gestureID = -1
            self.lefthand_gesture = "Null"
            return
        model_input = np.array([lefthand_landmarks])
        model_input = model_input.reshape(model_input.shape[0],-1)
        model_input = self.lefthand_scaler.transform(model_input)
        input_tensor = torch.from_numpy(model_input).float().to(self.device)
        with torch.no_grad():
            probabilities,logits = self.lefthand_model(input_tensor)
            max_probability, gesture_id_tensor = torch.max(probabilities, dim=1)
            gesture_id = gesture_id_tensor.item()
            if (max_probability.item() < 30):
                gesture_id = 18
    
        self.lefthand_gesture = self.mapping[gesture_id]
        self.lefthand_gestureID = gesture_id

    def righthand_gesture_detect(self, righthand_landmarks):
        if (len(righthand_landmarks) == 0):
            self.righthand_gestureID = -1
            self.righthand_gesture = "Null"
            return
        model_input = np.array([righthand_landmarks])
        model_input = model_input.reshape(model_input.shape[0],-1)
        model_input = self.righthand_scaler.transform(model_input)
        input_tensor = torch.from_numpy(model_input).float().to(self.device)
        with torch.no_grad():
            probabilities,logits = self.righthand_model(input_tensor)
            max_probability, gesture_id_tensor = torch.max(probabilities, dim=1)
            gesture_id = gesture_id_tensor.item()
            if (max_probability.item() < 30):
                gesture_id = 18
    
        self.righthand_gesture = self.mapping[gesture_id]
        self.righthand_gestureID = gesture_id

        


