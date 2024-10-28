import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from motpy import Detection, MultiObjectTracker

video_path = "Data/Thinh.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_video_path = "video2.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 3, frame_height))  # Output size for 3 frames

count = 0  
tracker = MultiObjectTracker(dt=0.1) 
object_ids = set()  

while cap.isOpened():
    ret, frame = cap.read()
    ret1, frame1 = cap.read()
    if not ret:
        print("Can't receive frame (end of video). Exiting ...")
        break

    # loc
    kernel_size = (25, 25)
    blurred_frame = cv2.GaussianBlur(frame, kernel_size, 40, 40)

    
    gray_frame = rgb2gray(blurred_frame)
    th = threshold_otsu(gray_frame)
    bin_img = gray_frame > th
    bin_img_uint8 = (bin_img * 255).astype(np.uint8)

    # Detection 
    contours, _ = cv2.findContours(bin_img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 450:
            x, y, w, h = cv2.boundingRect(contour)
            
            object_box = [x, y, x + w, y + h]
            detections.append(Detection(box=object_box))

            # ve box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # gray -> hsv
            obj_img = frame[y:y+h, x:x+w]
            hsv_obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)

            # cmp mau
            lower_red = np.array([0, 70, 50])
            upper_red = np.array([10, 255, 255])
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])

            
            mask_red = cv2.inRange(hsv_obj_img, lower_red, upper_red)
            mask_yellow = cv2.inRange(hsv_obj_img, lower_yellow, upper_yellow)
            mask_white = cv2.inRange(hsv_obj_img, lower_white, upper_white)

            # cmp 
            if np.sum(mask_red) > np.sum(mask_yellow) and np.sum(mask_red) > np.sum(mask_white):
                color = "Red"
            elif np.sum(mask_yellow) > np.sum(mask_red) and np.sum(mask_yellow) > np.sum(mask_white):
                color = "Yellow"
            else:
                color = "White"

            cv2.putText(frame, f'{color}', (x + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Update tracker with new detection
    tracker.step(detections=detections)
    tracks = tracker.active_tracks()

    # Check for new objects entering the right half of the frame
    for track in tracks:
        track_id = track.id
        x1, y1, x2, y2 = track.box

        if track_id not in object_ids and x2 > frame_width // 2:
            count += 1
            object_ids.add(track_id)

    # hien thi
    cv2.putText(frame, f'Count: {count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Dang Hung-Thinh_22119137', (frame.shape[1] - 470, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    bin_img_3ch = cv2.merge([bin_img_uint8, bin_img_uint8, bin_img_uint8])

    
    concatenated_frame = np.concatenate((frame, bin_img_3ch, frame1), axis=1)  

   
    cv2.imshow('Concatenated Frame', concatenated_frame)


    out.write(concatenated_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
