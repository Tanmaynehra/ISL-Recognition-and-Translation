import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
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

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    key1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    key2 = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([key1, key2, lh, rh])

actions = np.array(['अच्छा', 'अलविदा', 'घर', 'चतुर', 'धन्यवाद', 'नाम', 'भारतीय', 'मूर्ख', 'मेरा', 'लंबा', 'हेलो'])
no_sequences = 50
sequence_length = 10
model = load_model('slrhind.h5')
#print(model.summary())

sequence = []
sentence = []
predictions=[]
outp=[]
threshold = 0.6

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():


        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        keypoints = extract_keypoints(results)

        sequence.append(keypoints)
        sequence = sequence[-10:]
        
        if len(sequence) == 10:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-12:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            outp.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                        outp.append(actions[np.argmax(res)])

            if len(sentence) > 4: 
                sentence = sentence[-4:]

        
        
        
        font = ImageFont.truetype("C:/Users/Tanmay/AppData/Local/Microsoft/Windows/Fonts/NotoSans-Regular.ttf", 35)
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle([(0,435),(640,475)],fill="black")
        draw.text((3, 430), 'OUTPUT: '+' '.join(sentence), font=font,fill=(0,255,0))


        image = np.asarray(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
      
        cv2.imshow('Recognizing_sign_Hin', image)

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
    language = 'hi' 
    speech = gTTS(text =tx  ,lang = language, slow = False)
    speech.save('medium_1.mp3')
    os.system('start medium_1.mp3')