import PIL.Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys

'''
this model is from https://github.com/facebookresearch/segment-anything
to use this model,you can:
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
'''

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



image = cv2.imread('**')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()


sys.path.append("..")

sam_checkpoint = "***"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)




def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.7]])
        img[m] = color_mask
    ax.imshow(img)


def show_con(anns,img,blob=False):

    if len(anns) == 0:
        return
    img1 = img.copy()
    img2 = img.copy()
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    mask = np.zeros((img.shape[0],img.shape[1]))
    for ann in sorted_anns:
        m = ann['segmentation']
        mask[m] = 255
    mask = mask.astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel,iterations=10)  # 腐蚀图像
    mask = cv2.dilate(mask, kernel,iterations=10)  # 膨胀图像

    if blob:
        detector = cv2.SimpleBlobDetector()
        keypoints = detector.detect(mask)
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('img', im_with_keypoints)
        cv2.waitKey(0)

    a,_ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, a, -1, (0, 0, 255), 3)

    for contour in a:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        img1 = cv2.circle(img1, center, radius, (25, 0, 255), 2)
        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 0.5
        TEXT_THICKNESS = 1
        s = int(img[center[1],center[0],0]/3 + img[center[1],center[0],1]/3 + img[center[1],center[0],2]/3)
        TEXT = str(s)
        text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))
        cv2.putText(img1, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (252, 252, 252), TEXT_THICKNESS, cv2.LINE_AA)
        cv2.putText(img2, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (252, 252, 252), TEXT_THICKNESS, cv2.LINE_AA)

    cv2.drawContours(img2, a, -1, (0, 0, 255), 3)
    cv2.imshow('img',img1)
    cv2.waitKey(0)
    cv2.imshow('img', img2)
    cv2.waitKey(0)
    cv2.imwrite('***',img1)
    cv2.imwrite('***', img2)



mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=***,
    pred_iou_thresh=***,
    stability_score_thresh=***,
    crop_n_layers=***,
    crop_n_points_downscale_factor=***,
    min_mask_region_area=***,
)

masks2 = mask_generator_2.generate(image)



newmask  = list(filter(lambda x:x['area']>*** and x['area']<***,masks2))
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(newmask)
plt.axis('off')
plt.show()
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
show_con(newmask,image)
