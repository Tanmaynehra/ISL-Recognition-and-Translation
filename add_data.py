import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import augly.utils as utils
import augly.video as vidaugs
import random
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import save_model
from sklearn.metrics import accuracy_score

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keyp(results):
    key1 = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    key2 = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([key1, key2, lh, rh])

DATA_PATH = os.path.join('Dataset') 
actions = np.array(['bye','clever','foolish','good','hello','house','Indian','my','name','tall','thanks'] )
no_sequences = 40
sequence_length = 10
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 3  
num_frames = 10  
actions = np.array(['bye','clever','foolish','good','hello','house','Indian','my','name','tall','thanks'] )
num_videos_per_action = 40  
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  
    for action in actions:
        output_folder = os.path.join('AUG_TRY', action)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for i in range(num_videos_per_action):
            output_filename = os.path.join(output_folder, f'{action}_{i}.mp4')
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
            for j in range(num_frames):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)  
                if j == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, i), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(700)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, i), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)  
                out.write(frame)  
                if cv2.waitKey(100) & 0xFF == ord('q'):  
                    break
            out.release()
    cv2.destroyAllWindows()
cap.release()

'''VIDEOS_FOLDER = 'Dataset'
OUTPUT_FOLDER = 'AUGMENT'
NO_SEQUENCES = 40
SEQUENCE_LENGTH = 10
ACTIONS = ['bye','clever','foolish','good','hello','house','Indian','my','name','tall','thanks']
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        

        input_vid_path='{}/{}/{}_{}.mp4'.format(VIDEOS_FOLDER,action,action,sequence)
        n = random.randint(0, 2)
        output_vid_path='{}/{}/{}_{}.mp4'.format(OUTPUT_FOLDER,action,action,sequence+40)
        if(n==0):
            aug0=vidaugs.AddNoise()
            aug0(input_vid_path, output_vid_path)
        elif(n==1):

            aug1=vidaugs.Brightness(0.4)
            aug1(input_vid_path, output_vid_path)
        elif(n==2):

            aug2=vidaugs.HFlip()
            aug2(input_vid_path, output_vid_path)'''

DATA_PATH = os.path.join('Keypoints') 
ACTIONS = ['bye','clever','foolish','good','hello','house','Indian','my','name','tall','thanks']
NO_SEQUENCES = 80
for action in ACTIONS: 
    for sequence in range(NO_SEQUENCES):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

VIDEOS_FOLDER = 'Dataset'




ACTIONS = ['bye','clever','foolish','good','hello','house','Indian','my','name','tall','thanks']


NO_SEQUENCES = 80


SEQUENCE_LENGTH = 10

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        
        
        video_path = os.path.join(VIDEOS_FOLDER, action, f'{action}_{sequence}.mp4')
        cap = cv2.VideoCapture(video_path)

        
        for frame_num in range(SEQUENCE_LENGTH):

            
            ret, frame = cap.read()

            
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)            
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)


label_map = {label:num for num, label in enumerate(actions)}
seqs, lbl = [], []
for action in actions:
    for sequence in range(no_sequences):
        wdw = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            wdw.append(res)
        seqs.append(wdw)
        lbl.append(label_map[action])
X = np.array(seqs)
y = to_categorical(lbl).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12)
log_dir = os.path.join('allLogs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=False, activation='relu', input_shape=(10,1662)))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=25, callbacks=[tb_callback])
res = model.predict(X_test)
print("Output on Testing data: ")
print(actions[np.argmax(res[1])])
print(actions[np.argmax(y_test[1])])
yht = model.predict(X_test)
yt = np.argmax(y_test, axis=1).tolist()
yht = np.argmax(yht, axis=1).tolist()
print("Accuracy obtained: ",accuracy_score(yt, yht))
