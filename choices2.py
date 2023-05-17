import sys
import os
from tkinter import *
from easygui import *
from PIL import ImageTk, Image

def run():
    os.system('python speech_translator.py')
def run2():
    os.system('python speech_translator_hindi.py')
 
logo ='Audio.png'
disp="                       AUDIO TRANSLATION TO ISL"
choices = ["Translate Audio in English ","Translate Audio in Hindi" ,"Back","Exit"] 
reply   = buttonbox(disp,image=logo,choices=choices)
if reply ==choices[0]:
    run()
if reply == choices[1]:
    run2()
if reply ==choices[2]:
    os.system('choices.py')
    quit()
if reply == choices[3]:
    quit()
