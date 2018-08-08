import numpy as np
from skimage import segmentation, filters, img_as_ubyte
import cv2
import imutils
from imutils.video import FPS, WebcamVideoStream
import redis
import json

board_w=480
borad_h=640
board_w_mm=484.

train = cv2.imread("./template-2.jpg")
train_gray = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kpTrain, desTrain = sift.detectAndCompute(train_gray, None)

#Return a line based on two points.
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

# find intersection between two lines.
def find_intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def get_bw_img(image, threshold=150):
    wk_img = image.copy()
    
    # Apply threshold to grayscale img. (Converts to b&w)
    t, bw = cv2.threshold(wk_img, threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Global Threshold', bw)
    
    return bw.copy()

def remove_noise(image):
    wk_img=image.copy()
    
    # Baseline params
    kernel1=5
    kernel2=60
    max_iters = 10
    while True:
        kernel = np.ones((kernel1,kernel2))
        opened=cv2.morphologyEx(wk_img,cv2.MORPH_OPEN, kernel)    

        zero_count_l=np.count_nonzero(opened[:,:120])
        zero_count_r=np.count_nonzero(opened[:,1780:])
        if zero_count_l > 0 or zero_count_r > 0:
            #kernel1+=1 // maybe? for now let's not.
            kernel2+=1
        else:
            #print "Kernel:",(kernel1,kernel2)
            break
        max_iters -= 1
        if max_iters == 0:
            break

    # this cleans up the noise on the inside and reduces the number of
    # objects/contours.
    closed=cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((10,10)))
    
    return closed.copy()

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

def hide_surrounding_objects(img):
    mask = img > filters.threshold_otsu(img)
    return img_as_ubyte(segmentation.clear_border(mask))

rects = {
    "top": {
        "left": {
            "x": np.array([]),
            "y": np.array([])
        },
        "right": {
            "x": np.array([]),
            "y": np.array([])
        }
    },
    "bottom": {
        "left": {
            "x": np.array([]),
            "y": np.array([])
        },
        "right": {
            "x": np.array([]),
            "y": np.array([])
        }
    }
}
def find_homograpy_points(img_src):
    #todo: Split this guy out into functions
    wk_img = img_src.copy()

    gray = cv2.cvtColor(wk_img, cv2.COLOR_BGR2GRAY)
    bw_img = get_bw_img(image=gray, threshold=140)

    # remove everything outside of the board area
    success, rect = test(gray)
    if not success:
        return False, None

    rects["top"]["left"]["x"] = np.append(rects["top"]["left"]["x"], rect[0][0])
    rects["top"]["left"]["y"] = np.append(rects["top"]["left"]["y"], rect[0][1])
    rects["bottom"]["left"]["x"] = np.append(rects["bottom"]["left"]["x"], rect[1][0])
    rects["bottom"]["left"]["y"] = np.append(rects["bottom"]["left"]["y"], rect[1][1])
    rects["bottom"]["right"]["x"] = np.append(rects["bottom"]["right"]["x"], rect[2][0])
    rects["bottom"]["right"]["y"] = np.append(rects["bottom"]["right"]["y"], rect[2][1])
    rects["top"]["right"]["x"] = np.append(rects["top"]["right"]["x"], rect[3][0])
    rects["top"]["right"]["y"] = np.append(rects["top"]["right"]["y"], rect[3][1])

    window = 50
    rects["top"]["left"]["x"] = rects["top"]["left"]["x"][-window:]
    rects["top"]["left"]["y"] = rects["top"]["left"]["y"][-window:]
    rects["bottom"]["left"]["x"] = rects["bottom"]["left"]["x"][-window:]
    rects["bottom"]["left"]["y"] = rects["bottom"]["left"]["y"][-window:]
    rects["bottom"]["right"]["x"] = rects["bottom"]["right"]["x"][-window:]
    rects["bottom"]["right"]["y"] = rects["bottom"]["right"]["y"][-window:]
    rects["top"]["right"]["x"] = rects["top"]["right"]["x"][-window:]
    rects["top"]["right"]["y"] = rects["top"]["right"]["y"][-window:]

    average = np.array([
        [np.average(rects["top"]["left"]["x"]), np.average(rects["top"]["left"]["y"])],
        [np.average(rects["bottom"]["left"]["x"]), np.average(rects["bottom"]["left"]["y"])],
        [np.average(rects["bottom"]["right"]["x"]), np.average(rects["bottom"]["right"]["y"])],
        [np.average(rects["top"]["right"]["x"]), np.average(rects["top"]["right"]["y"])]
    ], dtype=np.int32)
    test_lined = cv2.polylines(gray.copy(),[average],True,255,3, cv2.LINE_AA)
    cv2.imshow('rect', test_lined)
    
    just_rect = np.zeros_like(gray)
    just_rect[min(rect[:,1]):max(rect[:,1]), min(rect[:,0]):max(rect[:,0])] = gray[min(rect[:,1]):max(rect[:,1]), min(rect[:,0]):max(rect[:,0])]

    bw_img = get_bw_img(image=just_rect, threshold=140)
    clean_img = remove_noise(bw_img)
    #cv2.imshow('clean', clean_img)

    edges = cv2.Canny(clean_img,1,1)
    cv2.imshow('debug', edges)

    edgePts=np.argwhere(edges)

    rect = cv2.minAreaRect(edgePts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # # convert box points to tuple
    rect_bot_left=tuple(box[0])
    rect_top_left=tuple(box[1])
    rect_top_right=tuple(box[2])
    rect_bot_right=tuple(box[3])

    # Fit triangle around edgepoints
    a, triangle = cv2.minEnclosingTriangle(np.array([edgePts]))

    # get proper points of the triangle...
    tri_t=triangle[0]
    tri_l=triangle[0]
    tri_r=triangle[0]

    for i in range(len(triangle)):
        if triangle[i][0][1] < tri_t[0][1]:
            tri_t=triangle[i]
          
        if triangle[i][0][0] < tri_l[0][0]:
            tri_l=triangle[i]
          
        if triangle[i][0][0] > tri_r[0][0]:
            tri_r=triangle[i]

    # Convert the points to tuple
    tri_bot_right=tuple(tri_r[0])
    tri_top=tuple(tri_t[0])
    tri_bot_left=tuple(tri_l[0])

    rect_top=line(rect_top_left, rect_top_right)
    tri_left=line(tri_bot_left, tri_top)
    tri_right=line(tri_bot_right, tri_top)

    l_intersection=find_intersection(rect_top, tri_left)
    r_intersection=find_intersection(rect_top, tri_right)

    if not l_intersection or not r_intersection:
        print("No intersections")
        return False, None

    homography_points = np.array([
        (int(tri_bot_left[0]), int(tri_bot_left[1])),
        (int(tri_bot_right[0]), int(tri_bot_right[1])),
        (int(r_intersection[0]), int(r_intersection[1])),
        (int(l_intersection[0]), int(l_intersection[1]))
    ])

    # Preview
    disp=working_img.copy()
    cv2.drawContours(disp,[box],0,(255,0,255),2)
    img = cv2.line(disp, tri_bot_left, tri_top, (255,255,0), 2)
    img = cv2.line(disp, tri_top, tri_bot_right, (255,255,0), 2)
    img = cv2.line(disp, tri_bot_right, tri_bot_left, (255,255,0), 2)
    img = cv2.polylines(disp, [homography_points], True, (255, 0, 0), 2)
    # cv2.imshow('edges', edges)
    cv2.imshow('Mapping', disp)

    return True, homography_points
        
def get_homographic_image(image, homography_points):
    wk_img=image.copy()

    pts_src = np.array(
        [
            homography_points[0],
            homography_points[1],
            homography_points[2],
            homography_points[3]
        ]
    )
    pts_dst = np.array(
        [
            [0, 639],
            [479, 639],
            [479, 0],
            [0, 0]        
        ]
    )

    h, status = cv2.findHomography(pts_src, pts_dst)
    warped = cv2.warpPerspective(wk_img, h, (board_w, borad_h))

    return warped.copy()

def find_puck(image):
    wk_img = image.copy()

    # conver to rgb, puck stands out better.
    rgb = cv2.cvtColor(wk_img, cv2.COLOR_BGR2RGB)

    pucklowerBound=np.array([0,0,110])
    puckupperBound=np.array([75,255,200])

    mask=cv2.inRange(rgb,pucklowerBound,puckupperBound)
    opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN, np.ones((1,3)))
    closed=cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((20,20)))

    #previews
    pts = np.argwhere(closed)
    if len(pts) > 0:
        center, radius = cv2.minEnclosingCircle(pts)

        return tuple(np.int0(center)), int(radius)
    else:
        print('Puck Not Detected')
        return (0,0), 0

def find_bot(image):
    wk_img = image.copy()

    hsv = cv2.cvtColor(wk_img, cv2.COLOR_RGB2HSV)
    # Don't care about top part of the board...
    hsv[0:200,:,:] = 0 

    botlowerBound=np.array([120,150,100])
    botupperBound=np.array([130,255,150])

    mask=cv2.inRange(hsv,botlowerBound,botupperBound)

    opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN, np.ones((10,5)))
    closed=cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((50,50)))

    #previews
    pts = np.argwhere(closed)
    if (len(pts) > 0):
        center, radius = cv2.minEnclosingCircle(pts)

        return tuple(np.int0(center)), int(radius)
    else:
        print('Bot Not Detected')
        return (0,0), 0

def set_puck_state(puck_pos):
    # make dictionary to be serialized to json
    p = json.loads("{\"x\":0,\"y\":0}")
    p['x'] = puck_pos[0]
    p['y'] = puck_pos[1]

    r.set("machine-state-puck", json.dumps(p))
    r.publish('state-changed', True)

def set_bot_state(bot_pos):
    # make dictionary to be serialized to json
    b = json.loads("{\"x\":0,\"y\":0}")
    b['x'] = int(bot_pos[0])
    b['y'] = int(bot_pos[1])

    r.set("machine-state-bot", json.dumps(b))
    r.publish('state-changed', True)

def set_board_state():
    # Put the board params in the state machine.
    board=json.loads("{\"pxw\":0,\"pxh\":0,\"mmw\":0}")
    board['pxw'] = board_w
    board['pxh'] = borad_h
    board['mmw'] = board_w_mm
    board['miny'] = 65

    r.set('machine-state', json.dumps(board))

vs = WebcamVideoStream(src=0)
vs = vs.start()
fps = FPS().start()
import time
start = time.time()
frames = 0
frameCt = 0
cur_fps=0
cur_h_points=np.asarray([])
counter = 0
#contours = np.vstack(contours).squeeze()

puck_center = -1
puck_radius = -1

bot_center = -1
bot_radius = -1

r=redis.StrictRedis(host='localhost',port=6379,db=0)
set_board_state()

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    img = imutils.resize(vs.read(), width=1024)
    if img is not None:
        cv2.imshow("raw", img)

        working_img = img.copy()

        h_points = np.asarray([])
        if frameCt < 50:
            success, h_points = find_homograpy_points(working_img)

        if frameCt % 1000 == 0:
            success, h_points = find_homograpy_points(working_img)
        if not success:
            continue

        if not cur_h_points.any():
            cur_h_points = h_points
        elif h_points.any():
            if not np.max(h_points - cur_h_points) < 50:
                cur_h_points = h_points
            else:
                counter += 1
                if counter > 100:
                    counter = 0
                    cur_h_points = np.mean( np.array([ cur_h_points, h_points ]), axis=0, dtype=np.int32)

        h_img = get_homographic_image(working_img, cur_h_points)            

        new_puck_center, new_puck_radius = find_puck(h_img)
        if puck_radius == -1 or ( new_puck_radius < puck_radius * 1.5 and new_puck_radius > puck_radius * 0.5):
            puck_radius = new_puck_radius
            puck_center = new_puck_center

        new_bot_center, new_bot_radius = find_bot(h_img)
        if bot_radius == -1 or ( new_bot_radius < bot_radius * 1.5 and new_bot_radius > bot_radius * 0.5):
            bot_radius = new_bot_radius
            bot_center = new_bot_center

        set_puck_state(puck_center)
        set_bot_state(bot_center)

        ###### Display Preview
        disp_img = h_img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        ftext='Frame: ' + str(frameCt)
        cv2.putText(disp_img,ftext,(10,50), font, 0.5,(255,0,255),2,cv2.LINE_AA)

        fps_text='FPS: ' + str(cur_fps)
        cv2.putText(disp_img,fps_text,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)

        puck_text='Puck: ' + str(puck_center[0]) + ', ' + str(puck_center[1])
        cv2.putText(disp_img,puck_text,(10,150), font, 0.5,(255,255,0),2,cv2.LINE_AA)

        bot_text='Bot: ' + str(bot_center[0]) + ', ' + str(bot_center[1])
        cv2.putText(disp_img,bot_text,(10,170), font, 0.5,(255,255,0),2,cv2.LINE_AA)

        img = cv2.circle(disp_img, puck_center, puck_radius, (0,255,0), 2)
        img = cv2.circle(disp_img, bot_center, bot_radius, (255,0,255), 2)

        cv2.imshow("Preview", disp_img)

        fps.update()
        frames += 1
        frameCt += 1
        
        if time.time() - start > 1:
            cur_fps = frames
            frames = 0
            start = time.time()
            print("FPS:", cur_fps)

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()