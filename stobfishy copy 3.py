import cv2
import numpy as np
import mss
import time
import pyautogui
import keyboard
import threading
import random
from ultralytics import YOLO

model = YOLO(r"E:\yolo8\runs\train\abbion_model9\weights\best.pt")
model.to('cuda') 
print("Model can detect these classes:", model.names)

bot_running = False
holding_left = False
holding_right = False

def bot_loop():
    global bot_running
    holding_left = False
    holding_right = False
    no_detect_start = None  # Track when no detection starts
    last_cast_time = 0      # Track last cast time

    with mss.mss() as sct:
        mon2 = sct.monitors[2]
        region = {
            "left": mon2['left'],
            "top": mon2['top'],
            "width": mon2['width'],
            "height": mon2['height']
        }

        while bot_running:
            now = time.time()
            img = np.array(sct.grab(region))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_resized = cv2.resize(img, (640, 640))  # Resize to model input size
            results = model(img_resized, conf=0.25)
            detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]

            if detected_classes:
                no_detect_start = None  # Reset timer if detection occurs

                # Hold left mouse while 'danger_left' is detected, release only on 'danger_right'
                if 'danger_left' in detected_classes and not holding_left:
                    pyautogui.mouseDown(button='left')
                    holding_left = True
                if 'danger_right' in detected_classes and holding_left:
                    pyautogui.mouseUp(button='left')
                    holding_left = False

                # Only detect 'wild' if 5 seconds have passed since last cast
                if 'wild' in detected_classes and (now - last_cast_time) >= 5:
                    pyautogui.click(button='left')
                    pyautogui.mouseUp(button='left')
                    holding_left = False

            else:
                # Ensure no mouse input before casting logic
                if holding_left:
                    pyautogui.mouseUp(button='left')
                    holding_left = False
                # No detection logic: cast (hold left mouse for 1 second, then release)
                if no_detect_start is None:
                    no_detect_start = now
                elapsed = now - no_detect_start
                # Only cast if "calm" is NOT detected and elapsed >= 1 second
                if elapsed >= 1 and 'calm' not in detected_classes:
                    pyautogui.mouseDown(button='left')
                    time.sleep(random.uniform(0.3, .8))  # Add randomness to hold time
                    pyautogui.mouseUp(button='left')
                    holding_left = False
                    last_cast_time = time.time()
                    no_detect_start = time.time() + 1  # Wait 5 seconds total after cast (1s hold + 4s wait)

            time.sleep(0.01)
        # cv2.destroyWindow("Bot View")  # Removed as it's not needed in the final script

def toggle_bot():
    global bot_running
    if not bot_running:
        bot_running = True
        threading.Thread(target=bot_loop, daemon=True).start()
        print("Bot started.")
    else:
        bot_running = False
        if holding_left:
            pyautogui.mouseUp(button='left')
        print("Bot stopped.")

keyboard.add_hotkey('ctrl+x', toggle_bot)

print("Press Ctrl+X to start/stop the bot.")
keyboard.wait()  # Keeps the script running

