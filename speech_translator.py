import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
import nltk
from nltk.stem import WordNetLemmatizer
import time

def func():
        r = sr.Recognizer()
        isl_gif=['good','morning','how','you','hello','thanks','careful','village','victory']
        stop_words=['is','are','am','to','has','have','had','it','by','for','be']
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        lemmatizer = WordNetLemmatizer()
        arr=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r', 's','t','u','v','w','x','y','z']
        with sr.Microphone() as source:
                
                r.adjust_for_ambient_noise(source) 
                i=0
                flg=0
                while True:
                        if flg==1:
                            break
                        print("Listening For Audio")
                        audio = r.listen(source)
                        
                        try:
                                ar=r.recognize_google(audio)
                                ar = ar.lower()
                                res = ar.split()
                                for ele in res:
                                    if ele in punc:
                                        res = res.replace(ele, "")
                                start_time = time.time()
                                print('You Said: ' + ar)
                                

                                for at in res:
                                    a = lemmatizer.lemmatize(at,pos='v')

                                    if(a.lower()=='stop'):
                                        print("Stop encoutered, listening is being terminated!")
                                        flg=1
                                        break
                                    elif(a.lower() in stop_words):
                                        continue        
                                    elif(a.lower() in isl_gif):
                                    
                                        cap= cv2.VideoCapture("ISL_Gifs/{0}.gif".format(a))
                                        if cap.isOpened() == False:
                                            print("Error File Not Found")
                                        while cap.isOpened():
                                            ret,frame= cap.read()
                                            if ret == True:
                                                cv2.namedWindow("Audio_Translator_Eng", cv2.WINDOW_NORMAL)
  
                                                cv2.resizeWindow("Audio_Translator_Eng", 400, 400)
                                            
                                                cv2.imshow('Audio_Translator_Eng', frame)
                                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                                    break
                                            else:
                                                cap.release()
                                                cv2.destroyAllWindows()
                                        latency = time.time() - start_time
                                        print('Latency: {:.2f} seconds'.format(latency))
                                    else:
                                        for i in range(len(a)):
                                            if(a[i] in arr):
                                            
                                                ImageAddress = 'letters/'+a[i]+'.jpg'
                                                ImageItself = Image.open(ImageAddress)
                                                ImageNumpyFormat = np.asarray(ImageItself)
                                                plt.imshow(ImageNumpyFormat)
                                                plt.draw()
                                                plt.pause(0.8)
                                            else:
                                                continue

                        except:
                            print(" ")
                        plt.close()
while 1:
  image='Audio.png'
  msg="                              AUDIO TRANSLATION"
  choices = ["Listen For Audio","Back","Exit"] 
  reply   = buttonbox(msg,image=image,choices=choices)
  if reply ==choices[0]:
        func()
  if reply == choices[1]:
        os.system('python main.py')
        quit()
  if reply == choices[2]:
        quit()
