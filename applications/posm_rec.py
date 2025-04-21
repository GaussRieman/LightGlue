import sys
sys.path.append("/datadrive/codes/opensource/features/LightGlue")

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import numpy as np
import cv2
import json
import time

# ALIKED+LightGlue
extractor_aliked = ALIKED(max_num_keypoints=1024).eval().cuda()  # load the extractor_aliked
matcher_aliked = LightGlue(features='aliked').eval().cuda()  # load the matcher_aliked

# SuperPoint+LightGlue
extractor_sp = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
matcher_sp = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
img0_path = "/datadrive/codes/opensource/features/LightGlue/data/posm/images/d4f8e549a5e2d42cdc30818e546192de.jpg"
# img0_path = "/datadrive/codes/opensource/features/LightGlue/applications/stitch.jpg"
img1_path = "/datadrive/codes/opensource/features/LightGlue/data/posm/POCM/4445852.png"

size = (320, 240)
image0 = load_image(img0_path, resize=size).cuda()
image1 = load_image(img1_path, resize=size).cuda()
print("image0: ", image0.shape)
print("image1: ", image1.shape)

t0 = time.time()
extractor = extractor_sp
matcher = matcher_sp

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
print("kp0: ", kpts0.shape)
print("kp1: ", kpts1.shape)

points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)


arr_points0 = points0.cpu().numpy()
arr_points1 = points1.cpu().numpy()

#find homography
src_pts = np.float32(arr_points1).reshape(-1, 1, 2)
dst_pts = np.float32(arr_points0).reshape(-1, 1, 2)
print("src_pts: ", src_pts.shape)
print("dst_pts: ", dst_pts.shape)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()
# print("matchesMask: ", matchesMask)
# calculate the number of inliers
inliers = 0
inlier_src_pts = []
inlier_dst_pts = []
for i in range(len(matchesMask)):
    if matchesMask[i] == 1:
        inliers += 1
        inlier_src_pts.append(src_pts[i])
        inlier_dst_pts.append(dst_pts[i])
print("inliers: ", inliers, inliers/len(matchesMask))
print("matched kp0:", inliers/kpts0.shape[0]) 
print("matched kp1:", inliers/kpts1.shape[0])

# draw the matches using plot_matches
axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(points0, points1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
viz2d.save_plot("matches.jpg", dpi=300)  # save the plot

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
viz2d.save_plot("kpts.jpg", dpi=300)  

t1 = time.time()
print("time: ", (t1 - t0))