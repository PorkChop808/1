import cv2
import numpy as np
import mss
import time
import pyautogui
import keyboard
import threading
import random
from ultralytics import YOLO

model = YOLO(r"E:\yolo8\runs\train\abbion_model10\weights\best.pt")
model.to('cuda')  # Use GPU if available
# model.half()  # Uncomment only if your GPU supports FP16

print("Model can detect these classes:", model.names)

bot_running = False
holding_left = False
bot_state = "ready"  # can be "ready", "waiting_for_wild", "danger_loop"

def bot_loop():
    global bot_running, holding_left, bot_state
    last_cast_time = 0
    danger_last_seen = None
    fish_count = 0  # Track how many times the full loop is completed

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
            img_resized = cv2.resize(img, (640, 640))
            results = model(img_resized, conf=0.10)  # Make sure this line is present
            detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]

            if bot_state == "ready":
                print(f"State: READY | Fish caught: {fish_count}")
                in_danger = ('danger_left' in detected_classes) or ('danger_right' in detected_classes)
                if not in_danger and (now - last_cast_time) >= 5:
                    hold_time = random.uniform(0.1, 0.8)
                    pyautogui.mouseDown(button='left')
                    time.sleep(hold_time)
                    pyautogui.mouseUp(button='left')
                    holding_left = False
                    last_cast_time = now
                    time.sleep(2)  # Wait 1 second before detecting 'wild'
                    bot_state = "waiting_for_wild"

            elif bot_state == "waiting_for_wild":
                print(f"State: WAITING_FOR_WILD | Fish caught: {fish_count}")
                if 'wild' in detected_classes:
                    pyautogui.click(button='left')
                    pyautogui.mouseUp(button='left')
                    holding_left = False
                    bot_state = "danger_loop"
                    danger_last_seen = now  # Start danger timer

            elif bot_state == "danger_loop":
                print(f"State: DANGER_LOOP | Fish caught: {fish_count}")
                in_danger = ('danger_left' in detected_classes) or ('danger_right' in detected_classes)
                if in_danger:
                    danger_last_seen = now  # Update last seen time
                    if not holding_left:
                            pyautogui.mouseDown(button='left')
                            holding_left = True
                    if 'danger_right' in detected_classes:
                        pyautogui.mouseUp(button='left')
                        holding_left = True
                    
                 
                else:
                    # Only leave danger loop if no danger detected for 2 seconds
                    if danger_last_seen and (now - danger_last_seen) >= 2:
                        if holding_left:
                            pyautogui.mouseUp(button='left')
                            holding_left = False
                        fish_count += 1  # Increment fish count after a full loop
                        bot_state = "ready"  # Enter ready state after leaving danger loop

            time.sleep(0.01)

def toggle_bot():
    global bot_running, holding_left
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
