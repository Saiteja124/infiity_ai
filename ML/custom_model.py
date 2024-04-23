from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

    model = YOLO("Object detection/custom_training.pt")
    classNames = ["dog", "person", "cat", "tv", "car", "meatballs", "marinara sauce",
                  "tomato soup", "chicken noodle soup", "french onion soup", "chicken breast",
                  "ribs", "pulled pork", "hamburger", "cavity", "Ac", "aeroplane", "lion",
                  "elephant", "zebra", "monkey", "tiger", "jiraffe", "fox", "apple", "knife",
                  "bag", "school bag", "table", "ball", "banana", "bat", "bed", "potted plant",
                  "frame", "lamp", "window", "beetroot", "bench", "trees", "bicycle", "helmet",
                  "boat", "ship", "book", "bottle gourd", "bottle", "hat", "watch", "mobile",
                  "saftey vest", "bus", "cabbage", "capsicum", "carrot", "cauliflower", "chair",
                  "laptop", "clock", "cock", "cooler", "deer", "doll", "dragon fruit", "duck",
                  "fan", "lights", "flute", "glass", "grocery store", "guitar", "gun", "helicopter",
                  "hen", "horse", "house", "bike", "jar", "kite", "light", "sofa", "mango",
                  "microwave oven", "tomato", "mushrooms", "onion", "orange", "parachute", "peacock",
                  "pen", "pencil", "sharpener", "eraser", "pine apple", "pomegranate", "potato",
                  "refrigerator", "ridge gourd", "skateboard", "stove", "strawberry", "cup",
                  "tooth bresh", "traffic lights", "truck", "umbrella", "utensils", "water tanker",
                  "watermelon", "auto", "keyboard", "pad", "scissor", "ipad", "monitor", "speds",
                  "mouse", "brick towel", "bulldozer", "concretemixer", "excavator", "handsaw",
                  "roadroller", "scissorlift", "towercrane", "woodenfloat"]
    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                color = (255, 0, 255)  # Default color: Magenta
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                if class_name == "Knife":
                    color = (0, 255, 255)  # Yellow color for Knife
                elif class_name == "Pistol":
                    color = (0, 255, 0)  # Green color for Pistol
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)  # Draw rectangle with specified color
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA) # Text color set to black
        yield img
        # out.write(img)
        # cv2.imshow("image", img)
        # if cv2.waitKey(1) & 0xFF==ord('1'):
        #    break
    # out.release()
    cv2.destroyAllWindows()
