import numpy as np
from skimage import segmentation, filters, img_as_ubyte
import cv2
import imutils
from imutils.video import FPS, WebcamVideoStream
import redis
import json
from lib import sift, rects

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

def hide_surrounding_objects(img):
    mask = img > filters.threshold_otsu(img)
    return img_as_ubyte(segmentation.clear_border(mask))

def find_homograpy_points(img_src):
    #todo: Split this guy out into functions
    wk_img = img_src.copy()

    gray = cv2.cvtColor(wk_img, cv2.COLOR_BGR2GRAY)

    # remove everything outside of the board area
    success, rect = sift.test(gray)
    if not success:
        return False, None

    rects.append(rect)
    test_lined = cv2.polylines(wk_img.copy(),[rects.average()],True,(255, 0, 0), 3, cv2.LINE_AA)
    #cv2.imshow('rect', test_lined)
    
    just_rect = np.zeros_like(gray)
    just_rect[min(rect[:,1]):max(rect[:,1]), min(rect[:,0]):max(rect[:,0])] = gray[min(rect[:,1]):max(rect[:,1]), min(rect[:,0]):max(rect[:,0])]

    bw_img = get_bw_img(image=just_rect, threshold=140)
    clean_img = remove_noise(bw_img)
    #cv2.imshow('clean', clean_img)

    edges = cv2.Canny(clean_img,1,1)
    #cv2.imshow('debug', edges)
    
    no_lava = np.zeros_like(edges)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(no_lava, contours, -1, (255,255,255), 3)

    line = np.zeros_like(no_lava)
    line[:, 500:501] = 1

    touches_line = []
    for contour in contours:
        touches = np.zeros_like(no_lava)
        cv2.drawContours(touches, [contour], -1, (255,255,255), 3)
        if np.any(touches & line):
            touches_line.append(contour)

    liners = np.zeros_like(no_lava)
    cv2.drawContours(liners, touches_line, -1, (255,255,255), 3)
    cv2.imshow('liners', liners)
    return False, None
    
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
        frames += 1
        frameCt += 1

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