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
#### 7. Converting to Grayscale and Applying Otsu's Threshold
```pytho
gray_frame = rgb2gray(blurred_frame)
th = threshold_otsu(gray_frame)
bin_img = gray_frame > th
bin_img_uint8 = (bin_img * 255).astype(np.uint8)
```
* Converts the image to grayscale.
* Uses **Otsu’s method** to determine a threshold for binarization.
* Converts the binary image to uint8 format for OpenCV processing.
---
#### 8. Detecting Objects (Contours)
```python
contours, _ = cv2.findContours(bin_img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
detections = []
```
* Finds contours (object boundaries) in the binary image.
```python
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 450:
        x, y, w, h = cv2.boundingRect(contour)
        object_box = [x, y, x + w, y + h]
        detections.append(Detection(box=object_box))
```
* Filters objects based on **area > 450** pixels.
* Extracts bounding box coordinates.
---
#### 9. Identifying Object Color
```python
obj_img = frame[y:y+h, x:x+w]
hsv_obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)
```
* Crops the detected object from the original frame.
* Converts the cropped image to HSV color space.
```python
lower_red = np.array([0, 70, 50])
upper_red = np.array([10, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])
```
* Defines HSV ranges for **red, yellow, and white** colors.
```python
mask_red = cv2.inRange(hsv_obj_img, lower_red, upper_red)
mask_yellow = cv2.inRange(hsv_obj_img, lower_yellow, upper_yellow)
mask_white = cv2.inRange(hsv_obj_img, lower_white, upper_white)
```
* Creates binary masks for each color.
```python
if np.sum(mask_red) > np.sum(mask_yellow) and np.sum(mask_red) > np.sum(mask_white):
    color = "Red"
elif np.sum(mask_yellow) > np.sum(mask_red) and np.sum(mask_yellow) > np.sum(mask_white):
    color = "Yellow"
else:
    color = "White"
```
* Determines the dominant color by comparing the sum of pixels in each mask.
```python
cv2.putText(frame, f'{color}', (x + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
```
* Displays the detected color on the frame.
---
#### 10. Object Tracking
```python
tracker.step(detections=detections)
tracks = tracker.active_tracks()
```
* Updates the tracker with new detections.
```python
for track in tracks:
    track_id = track.id
    x1, y1, x2, y2 = track.box
    if track_id not in object_ids and x2 > frame_width // 2:
        count += 1
        object_ids.add(track_id)
```
* Increments the counter when a new object crosses the right half of the frame.
#### 11. Displaying Information on the Frame
```python
cv2.putText(frame, f'Count: {count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(frame, f'Dang Hung-Thinh_22119137', (frame.shape[1] - 470, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
```
* Displays the count of tracked objects.
---
#### 12. Merging and Saving Frames
```python
bin_img_3ch = cv2.merge([bin_img_uint8, bin_img_uint8, bin_img_uint8])
concatenated_frame = np.concatenate((frame, bin_img_3ch, frame1), axis=1)
cv2.imshow('Concatenated Frame', concatenated_frame)
out.write(concatenated_frame)
```
* Merges three frames (original, binary, next frame) side by side.
* Displays and writes the frame to the output video.
#### 13. Releasing Resources
```python
if cv2.waitKey(25) & 0xFF == ord('q'):
    break
cap.release()
out.release()
cv2.destroyAllWindows()
```
* Pres `q` to exit
* Releases video resources.
---
### Result 
https://github.com/user-attachments/assets/2caed746-448c-493e-a6e8-890fcb4e2835

