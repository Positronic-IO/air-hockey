import numpy as np
from skimage import segmentation, filters, img_as_ubyte
import cv2
import imutils
from imutils.video import FPS, WebcamVideoStream
import redis
import json
from lib import sift, rects, triangles

board_w=480
borad_h=640
board_w_mm=484.

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
    kernel2=30
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

def hide_surrounding_objects(img):
    mask = img > filters.threshold_otsu(img)
    return img_as_ubyte(segmentation.clear_border(mask))

def find_homograpy_points(img_src):
    wk_img = img_src.copy()
    gray = cv2.cvtColor(wk_img, cv2.COLOR_BGR2GRAY)

    # find the bounding rect for the board
    success, rect = sift.test(gray)
    if not success:
        return False, None
    rects.append(rect)
    rect = rects.average() # smooth it out
    return True, rect

    test_lined = cv2.polylines(wk_img.copy(),[rect],True,(255, 0, 0), 3, cv2.LINE_AA)    
    just_rect = np.zeros_like(gray)
    just_rect[min(rect[:,1]):max(rect[:,1]), min(rect[:,0]):max(rect[:,0])] = gray[min(rect[:,1]):max(rect[:,1]), min(rect[:,0]):max(rect[:,0])]

    # find contours that intersect the midline
    bw_img = get_bw_img(image=just_rect, threshold=140)
    clean_img = remove_noise(bw_img)
    edges = cv2.Canny(clean_img,1,1)
    no_lava = np.zeros_like(edges)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(no_lava, contours, -1, (255,255,255), 3)
    midline = np.zeros_like(no_lava)
    mid = int(no_lava.shape[1]/2)
    midline[:, mid:mid+1] = 1
    touches_line = []
    for contour in contours:
        touches = np.zeros_like(no_lava)
        cv2.drawContours(touches, [contour], -1, (255,255,255), 3)
        if np.any(touches & midline):
            touches_line.append(contour)
    liners = np.zeros_like(no_lava)
    cv2.drawContours(liners, touches_line, -1, (255,255,255), 3)
    cv2.drawContours(test_lined, touches_line, -1, (255,255,255), 3)
    if not np.any(liners):
        return False, None

    # Fit triangle around contours
    a, triangle = cv2.minEnclosingTriangle(np.array([np.argwhere(liners.T)]))
    triangles.append(triangle)
    triangle = triangles.average() # smooth it out
    tri = np.zeros_like(no_lava)
    tri = cv2.polylines(test_lined, np.int32([triangle]), True, (0, 255, 0), 3)
    cv2.imshow('wrecked', test_lined)

    # find the points of intersection between rectangle & triangle
    rect_top = line( rect[0], rect[3])
    triangle_bottom_left = triangle[1]
    triangle_left = line(triangle[1], triangle[0])
    triangle_right = line(triangle[2], triangle[0])
    l_intersection = find_intersection(rect_top, triangle_left)
    r_intersection = find_intersection(rect_top, triangle_right)
    if not l_intersection or not r_intersection:
        print("No intersections")
        return False, None

    bl = rect[1] / 2 + triangle[1] / 2
    br = rect[2] / 2 + triangle[2] / 2
    homography_points = np.array([bl, br, r_intersection, l_intersection], dtype=int)
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

def find_puck(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pucklowerBound = np.array([0,0,130])
    puckupperBound = np.array([50,100,160])

    mask = cv2.inRange(rgb, pucklowerBound, puckupperBound)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.dilate(mask, el, iterations=3)

    _, contours,hierarchy = cv2.findContours(closed, 2, 1)
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (area > 30) ):
            contour_list.append(contour)
    if not contour_list:
        return (0, 0), 0
    c = max(contour_list, key = cv2.contourArea)
    center, radius = cv2.minEnclosingCircle(c)
    return tuple(np.int0(center)), int(radius)

def find_bot(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[0:200,:,:] = 0 
    lowerBound = np.array([0,150,100])
    upperBound = np.array([200,255,150])

    mask = cv2.inRange(hsv, lowerBound, upperBound)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.dilate(mask, el, iterations=3)

    _, contours,hierarchy = cv2.findContours(closed, 2, 1)
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (area > 30) ):
            contour_list.append(contour)
    if not contour_list:
        return (0, 0), 0
    c = max(contour_list, key = cv2.contourArea)
    center, radius = cv2.minEnclosingCircle(c)
    return tuple(np.int0(center)), int(radius)


def set_puck_state(puck_pos):
    # make dictionary to be serialized to json
    p = json.loads("{\"x\":0,\"y\":0}")
    p['x'] = int(puck_pos[0])
    p['y'] = int(puck_pos[1])

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
    if img is None:
        continue

    frames += 1
    frameCt += 1

    if frameCt < 25 or frameCt % 500 == 0:
        success, rect = find_homograpy_points(img)
    if not success:
        continue

    # cv2.polylines(disp, [cur_h_points], True, (255, 0, 0), 2)
    #cv2.imshow('preview', disp)

    #h_img = get_homographic_image(oper.copy(), cur_h_points)            
    
    board = np.zeros_like(img)
    board[min(rect[:,1]):max(rect[:,1]), min(rect[:,0]):max(rect[:,0])] = img[min(rect[:,1]):max(rect[:,1]), min(rect[:,0]):max(rect[:,0])]

    new_puck_center, new_puck_radius = find_puck(board)
    if new_puck_radius != 0:
        puck_center = new_puck_center
        puck_radius = new_puck_radius
        set_puck_state(puck_center)

    new_bot_center, new_bot_radius = find_bot(board)
    if new_bot_radius != 0:
        bot_center = new_bot_center
        bot_radius = new_bot_radius
        set_bot_state(bot_center)

    ###### Display Preview
    #disp_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    ftext='Frame: ' + str(frameCt)
    cv2.putText(img,ftext,(10,50), font, 0.5,(255,0,255),1,cv2.LINE_AA)

    fps_text='FPS: ' + str(cur_fps)
    cv2.putText(img,fps_text,(10,80), font, 0.5,(0,255,0),1,cv2.LINE_AA)

    if puck_radius != -1:
        puck_text='Puck: ' + str(puck_center[0]) + ', ' + str(puck_center[1]) + ', ' + str(puck_radius)
        cv2.circle(img, puck_center, puck_radius, (0,255,0), 2)
    else:
        puck_text='Puck: NOT FOUND'
    cv2.putText(img,puck_text,(10,150), font, 0.5,(255,255,0),1,cv2.LINE_AA)

    if bot_radius != -1:
        bot_text='Bot: ' + str(bot_center[0]) + ', ' + str(bot_center[1])
        cv2.circle(img, bot_center, bot_radius, (255,0,255), 2)
    else:
        bot_text='Bot: NOT FOUND'
    cv2.putText(img, bot_text, (10,170), font, 0.5,(255,255,0), 1, cv2.LINE_AA)

    cv2.polylines(img, [rect], True, (255, 0, 0), 1, cv2.LINE_AA)    

    cv2.imshow("Preview", img)

    fps.update()
    
    if time.time() - start > 1:
        cur_fps = frames
        frames = 0
        start = time.time()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()