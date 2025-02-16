### Block Diagram
![image](https://github.com/user-attachments/assets/79720cde-f0ea-40e0-8e72-2dde978bcdf0)

---
### Code Explanation
#### 1. Importing Required Libraries
```python
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from motpy import Detection, MultiObjectTracker
```
* **cv2**: Used for image and video processing.
* **numpy**: Supports matrix operations and fast numerical computations.
* **skimage.filters.threshold_otsu**: Otsu's thresholding method for binary image segmentation.
* **skimage.color.rgb2gray**: Converts an image to grayscale.
* **motpy**: A library for multi-object tracking.
---
#### 2. Opening the Video File and Checking for Errors
```python
video_path = "Data/Thinh.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()
```
* Opens the video file from the specified path.
* If the file cannot be opened, the program exits.
---
#### 3. Retrieving Video Properties and Creating an Output Video
```python
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_video_path = "video2.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 3, frame_height))  # Output size for 3 frames
```
* Gets video width, height, and frames per second (FPS).
* Creates an output video file (`video2.avi`) using the XVID codec.
* The output frame size is **3 times the original width** to accommodate three frames side by side.
---
#### 4. Initializing Counter and Object Tracker
```python
count = 0  
tracker = MultiObjectTracker(dt=0.1) 
object_ids = set()
```
* `count`: Keeps track of the number of detected objects that cross the frame.
* `tracker`: A multi-object tracker.
* `object_ids`: A set to store unique object IDs.
---
#### 5. Processing Video Frames
```python
while cap.isOpened():
    ret, frame = cap.read()
    ret1, frame1 = cap.read()
    if not ret:
        print("Can't receive frame (end of video). Exiting ...")
        break
```
* Reads frames from the video.
* If the video ends, the loop breaks.
#### 6. Applying Gaussian Blur for Noise Reduction
```python
kernel_size = (25, 25)
blurred_frame = cv2.GaussianBlur(frame, kernel_size, 40, 40)
```
* Uses **GaussianBlur** to smooth the image and reduce noise.
### Result 
https://github.com/user-attachments/assets/2caed746-448c-493e-a6e8-890fcb4e2835

