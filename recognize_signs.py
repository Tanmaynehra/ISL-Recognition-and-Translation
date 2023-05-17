import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from easygui import *
from gtts import gTTS

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results



def extract_kp(results):
    key1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    key2 = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([key1, key2, lh, rh])
actions = np.array(['bye', 'clever', 'foolish', 'good', 'hello', 'house', 'Indian', 'my', 'name', 'tall', 'thanks'])
no_sequences = 50
sequence_length = 10

model = load_model('slrmodel.h5')
#print(model.summary())

sequence = []
sentence = []
predictions=[]
outp=[]
thr = 0.6

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():


        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        print(results)
                      

        keypoints = extract_kp(results)

        sequence.append(keypoints)
        sequence = sequence[-10:]
        
        if len(sequence) == 10:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-12:])[0]==np.argmax(res):
                if res[np.argmax(res)] > thr: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            outp.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                        outp.append(actions[np.argmax(res)])

            if len(sentence) > 4: 
                sentence = sentence[-4:]

            
        
        cv2.rectangle(image, (0,440), (640, 480), (0, 0, 0), -1)
        cv2.putText(image, 'OUTPUT: '+' '.join(sentence), (3,465), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
      
        cv2.imshow('Recognizing Signs', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
message = "The recognized words were: "
msg= message + ', '.join(outp)
choices = ["Okay", "Listen"]
title = "Recognition"
output = ccbox(msg, title,choices)
if output:
    quit()
 

else:
    t=' '
    tx=t+', '.join(sentence)
    language = 'en' 
    speech = gTTS(text =tx  ,lang = language, slow = False)
    speech.save('medium_1.mp3')
    os.system('start medium_1.mp3')


