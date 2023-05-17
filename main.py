import sys
import os
from tkinter import *
from easygui import *
from PIL import ImageTk, Image

def run():
    os.system('python add_data.py')

def run2():
    os.system('python recognize.py')
def run3():
    os.system('python choices.py')
logo='logo.png'
disp="                        ISL RECOGNTION AND TRANSLATION"
choices = ["Add signs to the database", "Recognize Sign", "Translate Audio/Text to ISL", "Exit"] 
reply   = buttonbox(disp,image=logo,choices=choices)
if reply ==choices[0]:
    run()
if reply == choices[1]:
    run2()
if reply ==choices[2]:
    run3()
if reply == choices[3]:
    quit()

