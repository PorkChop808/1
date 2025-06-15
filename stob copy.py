import threading
import time
import keyboard
from PIL import ImageGrab
from ultralytics import YOLO
import numpy as np
import pyautogui
import pydirectinput
import random

model = YOLO(r"E:\yolo8\runs\train\abbion_model3\weights\best.pt")
print("Model can detect these classes:", model.names)

# Use a smaller screen region for faster capture (adjust as needed)
SCREEN_REGION = (0, 0, 3440, 1400)

bot_running = False
bot_thread = None

def detect_resources():
    screenshot = ImageGrab.grab(bbox=SCREEN_REGION)
    results = model.predict(
        screenshot,
        conf=0.5,
        device=0,
        imgsz=640,
        verbose=False
    )
    detected_objects = results[0].boxes.data.cpu().numpy()
    class_names = model.names
    resource_positions = {"roughstone": [], "cotton": []}
    for obj in detected_objects:
        x1, y1, x2, y2, conf, cls = obj
        cls = int(cls)
        class_name = class_names[cls].lower()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if class_name == "roughstone":
            resource_positions["roughstone"].append((center_x, center_y))
            print(f"Detected: roughstone at ({center_x}, {center_y}) conf={conf:.2f}")
        elif class_name == "cotton":
            resource_positions["cotton"].append((center_x, center_y))
            print(f"Detected: cotton at ({center_x}, {center_y}) conf={conf:.2f}")
    return resource_positions

def gather_resource(resource_positions, resource_type):
    if resource_positions:
        target_x, target_y = resource_positions[0]
        pyautogui.moveTo(target_x + SCREEN_REGION[0], target_y + SCREEN_REGION[1], duration=0.05)
        pydirectinput.click()
        if resource_type == "cotton":
            print("Moved to cotton. Waiting 15 seconds to allow for gathering.")
            time.sleep(15)
        else:
            time.sleep(1.5)

def move_quarter_screen_random():
    directions = [
        (SCREEN_REGION[2] // 4, 0),  # right
        (-SCREEN_REGION[2] // 4, 0), # left
        (0, SCREEN_REGION[3] // 4),  # down
        (0, -SCREEN_REGION[3] // 4), # up
    ]
    dx, dy = random.choice(directions)
    center_x = SCREEN_REGION[0] + SCREEN_REGION[2] // 2
    center_y = SCREEN_REGION[1] + SCREEN_REGION[3] // 2
    target_x = center_x + dx
    target_y = center_y + dy
    print(f"No resources found. Moving to ({target_x}, {target_y})")
    # Press spacebar to mount up before moving
    pydirectinput.press('space')
    time.sleep(0.2)
    pyautogui.moveTo(target_x, target_y, duration=0.2)
    pydirectinput.click()
    time.sleep(1.5)

def bot_loop():
    global bot_running
    while bot_running:
        detections = detect_resources()
        print(f"Detected resources at: {detections}")
        if detections["roughstone"]:
            gather_resource(detections["roughstone"], "roughstone")
        elif detections["cotton"]:
            gather_resource(detections["cotton"], "cotton")
        else:
            move_quarter_screen_random()

def toggle_bot():
    global bot_running, bot_thread
    if not bot_running:
        print("Bot started! Press Ctrl+X again to stop.")
        bot_running = True
        bot_thread = threading.Thread(target=bot_loop)
        bot_thread.start()
    else:
        print("Bot stopping...")
        bot_running = False
        if bot_thread is not None:
            bot_thread.join()

print("Press Ctrl+X to start/stop the bot. Press ESC to exit.")
keyboard.add_hotkey('ctrl+x', toggle_bot)

keyboard.wait('esc')
print("Exiting.")

