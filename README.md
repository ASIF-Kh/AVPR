# Automatic Vehicle and License Plate Recognition

## Overview

This project implements an automatic vehicle and license plate recognition system using YOLO object detection and OCR technology. The system can process both images and videos to detect vehicles, identify license plates, and extract the plate number using Optical Character Recognition (OCR).

## Implementation Details

The system uses:

- **YOLOv8** for vehicle and license plate detection.
- **EasyOCR** for extracting text from detected license plates.
- **OpenCV** for image and video processing.
- **Tkinter** for a simple GUI to select images or videos for processing.

### Components

1. **Vehicle Detection**:

   - Uses YOLOv8 model (`yolov8n.pt`) to detect vehicles.
   - Supports class IDs: car, motorcycle, bus, and truck.

2. **License Plate Detection**:

   - Uses a custom YOLO model (`license_plate_detector.pt`) to locate license plates.

3. **License Plate Recognition**:

   - Extracts the detected license plate region.
   - Converts it to grayscale and applies thresholding.
   - Uses EasyOCR to read text from the processed plate.
   - Filters the recognized text using regex patterns.

4. **Graphical User Interface (GUI)**:

   - Built with Tkinter.
   - Provides buttons to select images or videos for processing.

## Dataset Details

The models are trained on:

- **Vehicle Detection Model**: Trained on COCO dataset.
- **License Plate Detection Model**: Trained on a dataset of annotated license plate images.
- **OCR Model**: EasyOCR, which supports multiple languages and is pre-trained for English (`en`).

### Dataset Link

You can find the dataset for license plate detection [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/6).

## How to Run

### Prerequisites

Ensure you have the required dependencies installed:

```sh
pip install ultralytics opencv-python numpy easyocr tkinter
```

### Steps to Run

1. Clone the repository and navigate to the project folder.
2. Ensure you have the YOLO models (`yolov8n.pt` and `license_plate_detector.pt`) in the `models/` directory.
3. Run the script:
   ```sh
   python app.py
   ```
4. A GUI will appear with options to select an image or video.
5. Click "Process Image" or "Process Video" and select a file.
6. The processed output will be displayed with detected vehicles, license plates, and recognized text.

## Controls

- Press **'q'** to exit the video processing window.
- Close the image display window to proceed with another selection.

## Output

- Bounding boxes around detected vehicles and license plates.
- Recognized license plate text displayed on the image/video.
- Console output showing the extracted license plate number.

### Example Images

#### License Plate Recognition Output


https://github.com/user-attachments/assets/9437bf4a-42ce-43d5-a841-695c6c3fc99d


## Notes

- GPU acceleration is enabled if available (`gpu=True` in EasyOCR).
- The model is optimized for English license plates but can be fine-tuned for other formats.
- Ensure the video/image is clear for better recognition results.
