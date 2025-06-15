import cv2
import numpy as np
import mss
import time
import logging
import pyautogui
import keyboard
import threading
from ultralytics import YOLO

model = YOLO(r"E:\yolo8\runs\train\abbion_model7\weights\best.pt")
print("Model can detect these classes:", model.names)

bot_running = False
holding_left = False
holding_right = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stobfishy.log")
    ]
)

def bot_loop():
    global bot_running
    holding_left = False
    holding_right = False
    with mss.mss() as sct:
        region = sct.monitors[2]  # Capture the whole of monitor 2
        cv2.namedWindow("Bot View", cv2.WINDOW_NORMAL)
        # Optionally move the window to monitor 1
        mon1 = sct.monitors[1]
        win_x = mon1['left'] + 100
        win_y = mon1['top'] + 100
        cv2.moveWindow("Bot View", win_x, win_y)

        while bot_running:
            t0 = time.time()
            img = np.array(sct.grab(region))
            t1 = time.time()
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_resized = cv2.resize(img, (640, 640))  # Resize to model input size
            t2 = time.time()
            results = model(img_resized, conf=0.25)
            t3 = time.time()
            detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
            t4 = time.time()
            logging.info(f"TIMINGS: grab={t1-t0:.2f}s, prep={t2-t1:.2f}s, infer={t3-t2:.2f}s, post={t4-t3:.2f}s")
            logging.info(f"Detected classes: {detected_classes}")

            cv2.imshow("Bot View", img_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if detected_classes:
                if 'danger_left' in detected_classes and not holding_left:
                    pyautogui.mouseDown(button='left')
                    holding_left = True
                    logging.info("Mouse left button held down (danger_left detected).")
                elif 'danger_right' in detected_classes and holding_left:
                    pyautogui.mouseUp(button='left')
                    holding_left = False
                    logging.info("Mouse left button released (danger_right detected).")
            else:
                if holding_left:
                    pyautogui.mouseUp(button='left')
                    holding_left = False
                    logging.info("Mouse left button released (no detections).")
            time.sleep(0.01)
        cv2.destroyWindow("Bot View")

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

