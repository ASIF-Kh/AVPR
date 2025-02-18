import cv2
import numpy as np
from ultralytics import YOLO
from util import read_license_plate
import tkinter as tk
from tkinter import filedialog

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Draws decorative corner lines on the image.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left corner
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    
    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    
    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    
    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    
    return img

# Load YOLO models for vehicles and license plates
vehicle_model = YOLO('./models/yolov8n.pt')            # For vehicle detection
license_plate_detector = YOLO('./models/license_plate_detector.pt')  # For license plate detection
vehicle_classes = [2, 3, 5, 7] # List of vehicle class IDs to consider (e.g., car, motorcycle, bus, truck)


def process_video(video_path):
# Open video capture
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ----------------------------
        # 1. Run Vehicle Detection
        # ----------------------------
        vehicle_results = vehicle_model(frame)[0]
        vehicles_list = []  # To store vehicle bounding boxes (x1, y1, x2, y2)
        
        for detection in vehicle_results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicle_classes:
                # Convert coordinates to integers
                vx1, vy1, vx2, vy2 = int(x1), int(y1), int(x2), int(y2)
                vehicles_list.append((vx1, vy1, vx2, vy2))
                # Draw a decorative border around the detected vehicle
                draw_border(frame, (vx1, vy1), (vx2, vy2), color=(0, 255, 0), thickness=5, line_length_x=50, line_length_y=50)
                # Optionally, label the detection (e.g., "Vehicle")
                cv2.putText(frame, "Vehicle", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ----------------------------
        # 2. Run License Plate Detection & Associate with a Vehicle
        # ----------------------------
        lp_results = license_plate_detector(frame)[0]
        
        for lp_det in lp_results.boxes.data.tolist():
            lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = lp_det
            lp_x1, lp_y1, lp_x2, lp_y2 = int(lp_x1), int(lp_y1), int(lp_x2), int(lp_y2)
            
            # Draw a red rectangle around the detected license plate
            cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 2)
            
            # Compute the center point of the license plate detection
            lp_center_x = (lp_x1 + lp_x2) // 2
            lp_center_y = (lp_y1 + lp_y2) // 2
            
            # Find the first vehicle whose bounding box contains the license plate center
            associated_vehicle = None
            for (vx1, vy1, vx2, vy2) in vehicles_list:
                if vx1 <= lp_center_x <= vx2 and vy1 <= lp_center_y <= vy2:
                    associated_vehicle = (vx1, vy1, vx2, vy2)
                    break

            if associated_vehicle is None:
                # If no associated vehicle is found, skip overlaying the license plate crop.
                continue

            # ----------------------------
            # 3. Process and Overlay License Plate Crop
            # ----------------------------
            # Crop the license plate region from the frame
            lp_crop = frame[lp_y1:lp_y2, lp_x1:lp_x2].copy()
            lp_crop_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
            _, lp_crop_thresh = cv2.threshold(lp_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            if lp_crop.size == 0:
                continue

            # Resize the license plate crop to a fixed height while preserving aspect ratio.
            fixed_height = 400
            crop_height = lp_y2 - lp_y1
            if crop_height == 0:
                continue
            aspect_ratio = (lp_x2 - lp_x1) / crop_height
            fixed_width = int(fixed_height * aspect_ratio)
            lp_crop_resized = cv2.resize(lp_crop, (fixed_width, fixed_height))

            # Compute overlay position: center the license crop above the associated vehicle bounding box.
            vx1, vy1, vx2, vy2 = associated_vehicle
            overlay_x = (vx1 + vx2) // 2 - fixed_width // 2
            overlay_y = vy1 - fixed_height + 10  # 10-pixel gap above the vehicle
            if overlay_y < 0:
                overlay_y = 0

            # Ensure the overlay region is within frame boundaries
            if (overlay_y + fixed_height <= frame.shape[0] and
                overlay_x >= 0 and overlay_x + fixed_width <= frame.shape[1]):
                frame[overlay_y:overlay_y + fixed_height, overlay_x:overlay_x + fixed_width] = lp_crop_resized

            # Optionally, if you have an OCR function (e.g., read_license_plate), you can run it here and overlay text.
            # Example:
            license_text, license_score = read_license_plate(lp_crop_thresh)
            if license_text is not None:
                cv2.putText(frame, license_text, (overlay_x, overlay_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)

        # ----------------------------
        # 4. Display the Annotated Frame
        # ----------------------------
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Live Video", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image(img_path):
# Open video capture
    image = cv2.imread(img_path)
    frame = image

            # ----------------------------
            # 1. Run Vehicle Detection
            # ----------------------------
    vehicle_results = vehicle_model(frame)[0]
    vehicles_list = []  # To store vehicle bounding boxes (x1, y1, x2, y2)
            
    for detection in vehicle_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicle_classes:
            # Convert coordinates to integers
            vx1, vy1, vx2, vy2 = int(x1), int(y1), int(x2), int(y2)
            vehicles_list.append((vx1, vy1, vx2, vy2))
            # Draw a decorative border around the detected vehicle
            draw_border(frame, (vx1, vy1), (vx2, vy2), color=(0, 255, 0), thickness=5, line_length_x=50, line_length_y=50)
            # Optionally, label the detection (e.g., "Vehicle")
            cv2.putText(frame, "Vehicle", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ----------------------------
            # 2. Run License Plate Detection & Associate with a Vehicle
            # ----------------------------
    lp_results = license_plate_detector(frame)[0]
            
    for lp_det in lp_results.boxes.data.tolist():
        lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = lp_det
        lp_x1, lp_y1, lp_x2, lp_y2 = int(lp_x1), int(lp_y1), int(lp_x2), int(lp_y2)
                
        # Draw a red rectangle around the detected license plate
        cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 2)
                
        # Compute the center point of the license plate detection
        lp_center_x = (lp_x1 + lp_x2) // 2
        lp_center_y = (lp_y1 + lp_y2) // 2
                
        # Find the first vehicle whose bounding box contains the license plate center
        associated_vehicle = None
        for (vx1, vy1, vx2, vy2) in vehicles_list:
            if vx1 <= lp_center_x <= vx2 and vy1 <= lp_center_y <= vy2:
                associated_vehicle = (vx1, vy1, vx2, vy2)
                break

        if associated_vehicle is None:
            # If no associated vehicle is found, skip overlaying the license plate crop.
            continue

                # ----------------------------
                # 3. Process and Overlay License Plate Crop
                # ----------------------------
                # Crop the license plate region from the frame
        lp_crop = frame[lp_y1:lp_y2, lp_x1:lp_x2].copy()
        lp_crop_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
        _, lp_crop_thresh = cv2.threshold(lp_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        if lp_crop.size == 0:
            continue

                # Resize the license plate crop to a fixed height while preserving aspect ratio.
        fixed_height = 100
        crop_height = lp_y2 - lp_y1
        if crop_height == 0:
            continue
        aspect_ratio = (lp_x2 - lp_x1) / crop_height
        fixed_width = int(fixed_height * aspect_ratio)
        lp_crop_resized = cv2.resize(lp_crop, (fixed_width, fixed_height))

                # Compute overlay position: center the license crop above the associated vehicle bounding box.
        vx1, vy1, vx2, vy2 = associated_vehicle
        overlay_x = (vx1 + vx2) // 2 - fixed_width // 2
        overlay_y = vy1 - fixed_height - 10  # 10-pixel gap above the vehicle
        if overlay_y < 0:
            overlay_y = 0

                # Ensure the overlay region is within frame boundaries
        if (overlay_y + fixed_height <= frame.shape[0] and
            overlay_x >= 0 and overlay_x + fixed_width <= frame.shape[1]):
            frame[overlay_y:overlay_y + fixed_height, overlay_x:overlay_x + fixed_width] = lp_crop_resized

                # Optionally, if you have an OCR function (e.g., read_license_plate), you can run it here and overlay text.
                # Example:
        license_text, license_score = read_license_plate(lp_crop_thresh)
        if license_text is not None:
            cv2.putText(frame, license_text, (overlay_x, overlay_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)

            # ----------------------------
            # 4. Display the Annotated Frame
            # ----------------------------
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Detected Nameplate", display_frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

# GUI using Tkinter
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

def open_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_video(file_path)



root = tk.Tk()
root.title("Automatic Nameplate Recognition")

btn_image = tk.Button(root, text="Process Image", command=open_file)
btn_image.pack(pady=10)

btn_video = tk.Button(root, text="Process Video", command=open_video)
btn_video.pack(pady=10)


root.mainloop()