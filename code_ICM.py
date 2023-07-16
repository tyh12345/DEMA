import cv2
import numpy as np

color_blob = (0, 0, 255) # (b, g, r)
color_txt = (0, 0, 0)
src = '***'
output = '***'
# Read image
im_org = cv2.imread(src)
im_grey = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
# Resize
height = len(im_org[0, :]) ; width = len(im_org[:, 0])
im_grey_resize = cv2.resize(im_grey, (1000, int(1000*width/height)), interpolation=cv2.INTER_LINEAR_EXACT)
im_org_resize = cv2.resize(im_org, (1000, int(1000*width/height)), interpolation=cv2.INTER_LINEAR_EXACT)
# Gaussian Blurs
im = cv2.GaussianBlur(im_grey_resize, (1, 1), 20)
im_show = cv2.GaussianBlur(im_org_resize, (1, 1), 20)

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 255
params.thresholdStep = 1
params.filterByColor = False
params.filterByArea = True
params.minArea = 9000
params.maxArea = 13000
params.filterByCircularity = True
params.minCircularity = 0.4
params.filterByInertia = True
params.minInertiaRatio = 0.6
params.filterByConvexity = True
params.minConvexity = 0.7
params.maxConvexity = 0.99

detector = cv2.SimpleBlobDetector_create(params)
key_points = detector.detect(im)
# Extraction and Calculation
key_points_pt = np.zeros([len(key_points),2])
mask = np.zeros(im.shape, np.uint8)
for i in range(len(key_points)):
    # make mask
    center = (int(key_points[i].pt[0]), int(key_points[i].pt[1]))
    radius = int(0.5 * key_points[i].size)
    cv2.circle(mask, center, radius, 255, -1)
    where = np.where(mask == 255)
    intensity_values_from_original = im[where[0], where[1]]
    value = np.median(intensity_values_from_original)
    print (value)
    # Visualization
    data =  str(value)
    cv2.putText(im_show, data, (center[0]-45, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_txt, 2)
draw_image = cv2.drawKeypoints(im_show, key_points, np.array([]), color=color_blob, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show
cv2.imshow("key_points", draw_image)
cv2.imwrite(output, draw_image)
cv2.waitKey(0)




