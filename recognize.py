import sys
import os
from tkinter import *
from easygui import *
from PIL import ImageTk, Image

def run():
    os.system('python recognize_signs.py')
def run2():
    os.system('python recognize_signs_hin.py')
 

logo='Recognition.png'
disp="                                RECOGNIZE SIGNS"
choices = ["TRANSLATE ISL TO ENGLISH","TRANSLATE ISL TO HINDI","Back","Exit"] 
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
