# Arda Mavi
import os
import sys
import platform
import numpy as np
from time import sleep
from PIL import ImageGrab
from game_control import *
from predict import predict
from scipy.misc import imresize
from game_control import get_id
from get_dataset import save_img
import multiprocessing
from keras.models import model_from_json
from pynput.mouse import Listener as mouse_listener
from pynput.keyboard import Listener as key_listener
import time

def get_screenshot():
    img = ImageGrab.grab()
    img = np.array(img)[:,:,:3] # Get first 3 channel from image as numpy array.
    img = imresize(img, (150, 150, 3)).astype('float32')/255.
    return img

def save_event_keyboard(data_path, event, key):
    key = get_id(key)
    timestamp = int(time.time())
    data_path = data_path + '/{0}/{1}'.format(key, event)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    screenshot = get_screenshot()
    save_img(screenshot, data_path+ '/{0}.png'.format(timestamp))
    return

def save_event_mouse(data_path, x, y):
    data_path = data_path + '/{0},{1},0,0'.format(x, y)
    screenshot = get_screenshot()
    save_img(screenshot, data_path)
    return

def listen_keyboard():
    data_path = 'Data/Train_Data/Keyboard'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def on_press(key):
        try: k = key.char # single-char keys
        except: k = key.name # other keys
        save_event_keyboard(data_path, 1, k)

    def on_release(key):
        try: k = key.char # single-char keys
        except: k = key.name # other keys
        save_event_keyboard(data_path, 2, k)

    with key_listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def main():
    dataset_path = 'Data/Train_Data/'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Start to listening mouse with new process:
    listen_keyboard()
    return

if __name__ == '__main__':
    main()
