import threading
import time
import keyboard
from PIL import ImageGrab
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model (use your trained weights)
model = YOLO(r"E:\yolo8\runs\train\abbion_model2\weights\best.pt")

# Screen capture settings (smaller region for speed, or keep as is)
SCREEN_REGION = (0, 0, 1720, 720)  # Try a smaller region for faster processing

bot_running = False
bot_thread = None

def show_preview(screenshot, detected_objects, class_names):
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for obj in detected_objects:
        x1, y1, x2, y2, conf, cls = obj
        cls = int(cls)
        label = class_names[cls]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    # Resize to 1/5th original size for preview
    small_img = cv2.resize(img, (img.shape[1] // 5, img.shape[0] // 5))
    cv2.imshow("YOLOv8 Preview", small_img)
    cv2.moveWindow("YOLOv8 Preview", 3500, 100)

def detect_resources():
    screenshot = ImageGrab.grab(bbox=SCREEN_REGION)
    # Use a smaller imgsz for faster prediction
    results = model.predict(screenshot, conf=0.5, device=0, imgsz=320, verbose=False)
    detected_objects = results[0].boxes.data.cpu().numpy()
    class_names = model.names
    show_preview(screenshot, detected_objects, class_names)
    gather_classes = {"cotton", "copper", "roughstone"}
    attack_class = "impala"
    resource_positions = []
    impala_positions = []
    for obj in detected_objects:
        x1, y1, x2, y2, conf, cls = obj
        cls = int(cls)
        class_name = class_names[cls].lower()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if class_name in gather_classes:
            resource_positions.append((center_x, center_y))
        elif class_name == attack_class:
            impala_positions.append((center_x, center_y))
    return resource_positions, impala_positions

def bot_loop():
    global bot_running
    while bot_running:
        start = time.perf_counter()
        resources, impalas = detect_resources()
        print(f"Detected resources: {resources}, Detected impalas: {impalas}")
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # Print FPS
        elapsed = time.perf_counter() - start
        print(f"Loop time: {elapsed:.3f}s, FPS: {1/elapsed:.1f}")
    cv2.destroyAllWindows()

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
        print("Bot stopped. Press Ctrl+X to start again.")

print("Press Ctrl+X to start/stop the bot. Press ESC to exit.")
keyboard.add_hotkey('ctrl+x', toggle_bot)
keyboard.wait('esc')
print("Exiting.")
