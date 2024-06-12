from mss import mss
import cv2
import numpy as np
from time import time,sleep
import pyautogui
from imutils.object_detection import non_max_suppression
import keyboard

screen_width = 2560
screen_height = 1440

topleft = (1090,350)
bottom_right = (1475,1050)
dimensions = {
    'left': topleft[0],
    'top': topleft[1],
    'width': bottom_right[0]-topleft[0],
    'height': bottom_right[1]-topleft[1]
}

flake = cv2.imread('flake.jpg')
if len(flake.shape) == 3:
    flake = cv2.cvtColor(flake, cv2.COLOR_BGR2GRAY)

width = flake.shape[1]
height = flake.shape[0]
pyautogui.PAUSE = 0


def match_template(screen_gray, template, scale):
    resized_template = cv2.resize(template, (int(width * scale), int(height * scale)))
    result = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    locations = np.where(result >= threshold)
    positions = list(zip(*locations[::-1]))
    
    rectangles = []
    for pt in positions:
        rect = [int(pt[0]), int(pt[1]), int(pt[0] + resized_template.shape[1]), int(pt[1] + resized_template.shape[0])]
        rectangles.append(rect)
        rectangles.append(rect)
    return rectangles


fps_time = time()
sct = mss()
scales = np.arange(0.2, 0.6, 0.1)
offsetY = 100
while True:
    scr = np.array(sct.grab(dimensions))
    scr_gray = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
    
    all_rectangles = []
    for scale in scales:
        rectangles = match_template(scr_gray, flake, scale)
        all_rectangles.extend(rectangles)
    #print(all_rectangles)

    if all_rectangles:
        pick = non_max_suppression(np.array(all_rectangles), probs=None, overlapThresh=0.3)
        
        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(scr, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.circle(scr,((startX + int(abs((endX-startX)/2))), startY + int(abs((endY-startY)/2))),5,(255,0,0),2)
            if startY + int(abs((endY-startY)/2))+ topleft[1] > offsetY + topleft[1]:
                pyautogui.click((startX + int(abs((endX-startX)/2)))+ topleft[0], startY + int(abs((endY-startY)/2))+ topleft[1])


    cv2.imshow('Screen', scr)
    
    print('FPS: {}'.format(1 / (time() - fps_time)))
    fps_time = time()

    if (cv2.waitKey(1) & 0xFF) == ord('q') or keyboard.is_pressed('q'):
        cv2.destroyAllWindows()
        break

