### Block Diagram
![image](https://github.com/user-attachments/assets/79720cde-f0ea-40e0-8e72-2dde978bcdf0)
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
### Result 
https://github.com/user-attachments/assets/2caed746-448c-493e-a6e8-890fcb4e2835

