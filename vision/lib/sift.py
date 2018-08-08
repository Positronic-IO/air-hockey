import numpy as np
import cv2

train = cv2.imread("./template-2.jpg")
train_gray = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kpTrain, desTrain = sift.detectAndCompute(train_gray, None)

def test(test_gray):
    kpTest, desTest = sift.detectAndCompute(test_gray, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desTrain, desTest,k=2)
    except:
        return False, None

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 3
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kpTrain[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpTest[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = train_gray.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        if len(pts) == 0:
            return np.asarray([])
        try:
            dst = cv2.perspectiveTransform(pts,M)
        except:
            return False, None
        ret = np.int32(dst)[:,0,:]

        # at least 0
        ret[ret < 0] = 0
        th,tw = test_gray.shape
        
        # not to exceed width
        mask = np.zeros_like(ret)
        mask[:,0] = 1
        ret[(ret > tw) & (mask == 1)] = tw

        # not to exceed height
        mask = np.zeros_like(ret)
        mask[:,1] = 1
        ret[(ret > th) & (mask == 1)] = th
        return True, ret
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        return False, None
