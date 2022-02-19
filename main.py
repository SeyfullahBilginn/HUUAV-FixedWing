import numpy as np
import cv2


imageFrame = cv2.imread("./SampleImages/1.jpg")

cv2.imshow("OriginalImage", imageFrame)
# apply smoothing function to remove unnecessery noises
unNoised = cv2.fastNlMeansDenoisingColored(imageFrame, None, 20, 20, 14, 42)

# Convert the imageFrame in
# BGR(RGB color space) to
# HSV(hue-saturation-value)
# color space
hsvFrame = cv2.cvtColor(unNoised, cv2.COLOR_BGR2HSV)

# Set range for red color and
# define mask
red_lower = np.array([136, 87, 111], np.uint8)
red_upper = np.array([180, 255, 255], np.uint8)
red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

kernal = np.ones((10, 10), "uint8")

# For red color
red_mask = cv2.dilate(red_mask, kernal)

red_mask_not = cv2.bitwise_not(red_mask)
# Set our filtering parameters
# Initialize parameter settiing using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = False
params.minArea = 10

# Set Circularity filtering parameters
params.filterByCircularity = False
params.minCircularity = 0.1

# Set Convexity filtering parameters
params.filterByConvexity = False
params.minConvexity = 0.87

# # Set inertia filtering parameters
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(red_mask_not)

# Draw blobs on our image as green circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(imageFrame, keypoints, blank, (0, 255, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))

cv2.putText(blobs, text, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show blobs
cv2.imshow("Filtering Circular Blobs Only", blobs)

cv2.waitKey()
