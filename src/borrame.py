import pygame
import time

import cv2
import numpy as np
import sys
import threading

import ntpath
from time import sleep

import os
import os.path
from screeninfo import get_monitors

import glob

#pygame.init()
pygame.mixer.init()
sounda= pygame.mixer.Sound("sounds/chiquito/violent/chiquito-fuego.wav")

sounda.play()
time.sleep (2)
print ("END")