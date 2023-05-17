import sys
import os
from tkinter import *
from easygui import *
from PIL import ImageTk, Image

def run():
    os.system('python choices2.py')
def run2():
    os.system('python text_translator.py')
 
logo='Translation.png'
disp="                         AUDIO/TEXT TRANSLATION TO ISL"
choices = ["Translate Audio","Translate Text","Back","Exit"] 
reply   = buttonbox(disp,image=logo,choices=choices)
if reply ==choices[0]:
    run()
if reply == choices[1]:
    run2()
if reply ==choices[2]:
    os.system('python main.py')
    quit()
if reply == choices[3]:
    quit()
