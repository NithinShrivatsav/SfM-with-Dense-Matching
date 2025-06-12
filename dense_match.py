
# Dense Stereo Matching and 3D Reconstruction Pipeline

import numpy as np
import heapq
import cv2
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

def ReliableArea(im):
    diff_up = np.abs(im - np.roll(im, -1, axis=0))
    diff_down = np.abs(im - np.roll(im, 1, axis=0))
    diff_left = np.abs(im - np.roll(im, -1, axis=1))
    diff_right = np.abs(im - np.roll(im, 1, axis=1))
    rim = np.maximum.reduce([diff_up, diff_down, diff_left, diff_right])
    rim[0, :] = rim[-1, :] = rim[:, 0] = rim[:, -1] = 0
    rim = (rim < 0.01).astype(np.float64)
    return 1 - rim

def ZNCCpatch_all(im, HalfSizeWindow):
    H, W = im.shape
    win_size = 2 * HalfSizeWindow + 1
    pad = HalfSizeWindow
    padded_im = np.pad(im, pad_width=pad, mode='reflect')
    windows = sliding_window_view(padded_im, (win_size, win_size)).reshape(H, W, -1)
    zncc_mean = np.mean(windows, axis=2, keepdims=True)
    zncc_std = np.sqrt(np.sum(windows ** 2, axis=2, keepdims=True) - (win_size * zncc_mean) ** 2)
    zncc = (windows - zncc_mean) / (zncc_std + 1e-8)
    return zncc

class PriorityQueue:
    def __init__(self):
        self.heap = []
    def push(self, item, priority):
        heapq.heappush(self.heap, (-priority, item))
    def pop(self):
        priority, item = heapq.heappop(self.heap)
        return item, -priority
    def size(self):
        return len(self.heap)

def safe_minmax(rng):
    lst = list(rng)
    return (min(lst), max(lst)) if lst else (0, -1)

def propagate(initial_match, img1, img2, matchable_im_i, matchable_im_j, zncc_i, zncc_j, WinHalfSize):
    CostMax = 0.7
    match_im_i = np.stack([(matchable_im_i - 2)] * 2, axis=-1)
    match_im_j = np.stack([(matchable_im_j - 2)] * 2, axis=-1)
    maxMatchingNumber = min(np.count_nonzero(matchable_im_i), np.count_nonzero(matchable_im_j))
    match_heap = PriorityQueue()
    match_pair = initial_match.copy().tolist()
    print(len(match_pair))

    for i, match in enumerate(match_pair):
        x0, y0, x1, y1 = map(int, match[:4])
        cost = float(np.sum(zncc_i[x0, y0, :] * zncc_j[x1, y1, :]))
        if len(match) < 5:
            match.append(cost)
        match_heap.push(i, cost)

    while maxMatchingNumber > 0 and match_heap.size() > 0:
        bestIndex, _ = match_heap.pop()
        x0, y0, x1, y1 = map(int, match_pair[bestIndex][:4])
        x0r = range(max(WinHalfSize, x0 - WinHalfSize), min(matchable_im_i.shape[0] - WinHalfSize, x0 + WinHalfSize + 1))
        y0r = range(max(WinHalfSize, y0 - WinHalfSize), min(matchable_im_i.shape[1] - WinHalfSize, y0 + WinHalfSize + 1))
        x1r = range(max(WinHalfSize, x1 - WinHalfSize), min(matchable_im_j.shape[0] - WinHalfSize, x1 + WinHalfSize + 1))
        y1r = range(max(WinHalfSize, y1 - WinHalfSize), min(matchable_im_j.shape[1] - WinHalfSize, y1 + WinHalfSize + 1))
        x1min, x1max = safe_minmax(x1r)
        y1min, y1max = safe_minmax(y1r)
        local_heap = []
        for yy0 in y0r:
            for xx0 in x0r:
                if match_im_i[xx0, yy0, 0] == -1:
                    xx = xx0 + x1 - x0
                    yy = yy0 + y1 - y0
                    for yy1 in range(max(y1min, yy - 1), min(y1max, yy + 2)):
                        for xx1 in range(max(x1min, xx - 1), min(x1max, xx + 2)):
                            if match_im_j[xx1, yy1, 0] == -1:
                                cost = float(np.sum(zncc_i[xx0, yy0, :] * zncc_j[xx1, yy1, :]))
                                if 1 - cost <= CostMax:
                                    local_heap.append([xx0, yy0, xx1, yy1, cost])
        local_heap.sort(key=lambda x: -x[4])
        for best_local in local_heap:
            xx0, yy0, xx1, yy1, cost = best_local
            if match_im_i[xx0, yy0, 0] == -1 and match_im_j[xx1, yy1, 0] == -1:
                match_im_i[xx0, yy0, :] = [xx1, yy1]
                match_im_j[xx1, yy1, :] = [xx0, yy0]
                match_pair.append([xx0, yy0, xx1, yy1, cost])
                match_heap.push(len(match_pair) - 1, cost)
                maxMatchingNumber -= 1

    
    print("Initial matches:", len(initial_match))
    print("Total matches after propagation:", len(match_pair))

    new_matches = match_pair[len(initial_match):]
    return np.array(new_matches), match_im_i, match_im_j

def denseMatch(pair, frames, frameID_i, frameID_j):
    HalfSizePropagate = 2
    im1 = frames['images'][frameID_i]
    im2 = frames['images'][frameID_j]

    im1 = im1.astype(np.float64) / 256.0
    matchable_image1 = ReliableArea(im1)
    zncc1 = ZNCCpatch_all(im1, HalfSizePropagate)

    im2 = im2.astype(np.float64) / 256.0
    matchable_image2 = ReliableArea(im2)
    zncc2 = ZNCCpatch_all(im2, HalfSizePropagate)

    matches = pair['matches']
    initial_match = np.round(np.column_stack([
        im1.shape[0]/2 - matches[1, :],
        im1.shape[1]/2 - matches[0, :],
        im1.shape[0]/2 - matches[3, :],
        im1.shape[1]/2 - matches[2, :],
        np.zeros(matches.shape[1])
    ])).astype(int)
    initial_match = np.squeeze(initial_match)

    match_pair, _, _ = propagate(initial_match, im1, im2, matchable_image1, matchable_image2, zncc1, zncc2, HalfSizePropagate)
    
    if match_pair.size == 0:
        pair['denseMatch'] = np.zeros((5, 0))
        return pair

    match_pair = match_pair.astype(float)
    match_pair[:, 0] = im1.shape[0]/2 - match_pair[:, 0]
    match_pair[:, 1] = im1.shape[1]/2 - match_pair[:, 1]
    match_pair[:, 2] = im2.shape[0]/2 - match_pair[:, 2]
    match_pair[:, 3] = im2.shape[1]/2 - match_pair[:, 3]
    match_pair = match_pair.T
    match_pair = match_pair[[1, 0, 3, 2, 4], :]
    pair['denseMatch'] = match_pair
    return pair

def triangulate_point(p1, p2, M1, M2):
    A = np.array([
        (p1[0] * M1[2] - M1[0]),
        (p1[1] * M1[2] - M1[1]),
        (p2[0] * M2[2] - M2[0]),
        (p2[1] * M2[2] - M2[1])
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def create_dense_keypoints(img, step=4, size=8):
   kp = [cv2.KeyPoint(x, y, size)
         for y in range(0, img.shape[0], step)
         for x in range(0, img.shape[1], step)]
   return kp

def generate_sparse_matches(img1, img2):
    orb = cv2.ORB_create(nfeatures=5000)
    kp1 = create_dense_keypoints(img1)
    kp2 = create_dense_keypoints(img2)
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)
    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sparse_matches = np.vstack((pts1.T, pts2.T))

    # draw_params = dict(matchColor=(0, 255, 0),
    #               singlePointColor=None,
    #               flags=2)
    # img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, **draw_params)
    # # Show result
    # plt.figure(figsize=(15, 10))
    # plt.imshow(img_matches, cmap='gray')
    # plt.title("Dense SIFT Matching with RANSAC")
    # plt.axis('off')
    # plt.show()
    # plt.savefig("sparse_matches2.png")
    
    return sparse_matches

def draw_dense_matches_cv2(img1, img2, matches_5xN, plot_name):
   # Convert to BGR if grayscale
   if len(img1.shape) == 2:
       img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_GRAY2BGR)
   if len(img2.shape) == 2:
       img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_GRAY2BGR)
   keypoints1 = []
   keypoints2 = []
   dmatches = []
   for i in range(matches_5xN.shape[1]):
       y0, x0, y1, x1 = matches_5xN[0, i], matches_5xN[1, i], matches_5xN[2, i], matches_5xN[3, i]
       kp1 = cv2.KeyPoint(float(y0), float(x0), 1)
       kp2 = cv2.KeyPoint(float(y1), float(x1), 1)
       keypoints1.append(kp1)
       keypoints2.append(kp2)
       match = cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0)
       dmatches.append(match)
   match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, dmatches, None,
                               matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2)
   plt.figure(figsize=(12, 6))
   plt.imshow(match_img[..., ::-1])  # Convert BGR to RGB for display
   plt.axis('off')
   plt.title("Dense Matches Visualized with OpenCV")
   plt.show()
   plt.savefig(plot_name)

def plot_keypoints(img, matches_5xN, plot_name):
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Extract keypoints from matches_5xN
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    keypoints = []
    for i in range(matches_5xN.shape[1]):
        y, x = matches_5xN[0, i], matches_5xN[1, i]
        shifted_x = x - width // 2 + center_x
        shifted_y = y - height // 2 + center_y
        kp = cv2.KeyPoint(float(x), float(y), 1)
        keypoints.append(kp)
    
    # Plot the image and keypoints
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255), 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_with_keypoints[...,::-1])
    plt.axis('off')
    plt.title("Keypoints Visualized")
    plt.savefig(plot_name)

if __name__=="__main__":
    img1 = cv2.imread('/opt/smarts/sta/Src/python/demo/stanford_assignment/assignment_3/data/statue/images/B24.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('/opt/smarts/sta/Src/python/demo/stanford_assignment/assignment_3/data/statue/images/B25.jpg', cv2.IMREAD_GRAYSCALE)

    frames = {'images':[img1, img2], 'imsize':img1.shape}
    matches = generate_sparse_matches(img1, img2)
    matches = np.squeeze(matches)
    pair = {'matches': matches}
    dense_matches = denseMatch(pair, frames, 0, 1)
    print(dense_matches['denseMatch'].shape)
    draw_dense_matches_cv2(img1, img2, matches, 'mvs_sparse_matches.png')
    plot_keypoints(img1, matches, 'mvs_single_image.png')
    plot_keypoints(img2, matches, 'mvs_single_image2.png')
    draw_dense_matches_cv2(img1, img2, dense_matches['denseMatch'], 'mvs_dense_matches.png')

    plot_keypoints(img1, dense_matches['denseMatch'], 'mvs_single_image_dense1.png')
    plot_keypoints(img2, dense_matches['denseMatch'], 'mvs_single_image_dense2.png')


    dense_matches_gt = np.load('/opt/smarts/sta/Src/python/demo/stanford_assignment/assignment_3/data/statue/dense_matches.npy', 
                               allow_pickle=True, encoding='latin1')
    draw_dense_matches_cv2(img1, img2, dense_matches_gt[0], 'mvs_dense_matches_gt.png')
    plot_keypoints(img1, dense_matches_gt[3], 'mvs_single_image_dense1_gt.png')


