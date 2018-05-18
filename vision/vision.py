import numpy as np
import cv2
import imutils
from WebcamVideoSteam import WebcamVideoStream
from imutils.video import FPS
#import redis
import json

corner_lower=np.array([75,106,120])
corner_upper=np.array([90,210,200])

side_lower = np.array([47,47,110])
side_upper = np.array([62,112,174])

goal_lower = np.array([17,114,87])
goal_upper = np.array([26,168,111])


# Hard corded corner points
top_left = (636, 55)
top_right = (1213, 28)
bot_right = (1892 ,  866)
bot_left = (48, 949)

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
    
    # Convert image to grayscale
    gray = cv2.cvtColor(wk_img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to grayscale img. (Converts to b&w)
    t, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return bw.copy()

def remove_noise(image, kernel_open=None, kernel_close=None):
    wk_img=image.copy()
    
    # Baseline params
    if kernel_open is None:
        kernel_open = np.ones((10,30))

    if kernel_close is None:
        kernel_close = np.ones((10,10))

    # we only want large objects since we are trying to detect
    # the board. 1x60 is a good starting point. Can call again
    # with override if the result is not good enough.
    # this cleans up the outside noise
    opened=cv2.morphologyEx(wk_img,cv2.MORPH_OPEN, kernel_open)

    #cv2.imshow('Open', opened)
    
    # this cleans up the noise on the inside and reduces the number of
    # objects.
    closed=cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    #cv2.imshow('Close', closed)
    
    return closed.copy()

def get_contour_points(contours):
    pts = []
    for i in range(0, len(contours)):
        for j in range(0, len(contours[i])):
            pts.append(contours[i][j])

    return np.array(pts)

def get_board_shape(points):

    # first, get minarearect and get box points from the rect.
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # convert points to tuple
    rect_bot_left=tuple(box[0])
    rect_top_left=tuple(box[1])
    rect_top_right=tuple(box[2])
    rect_bot_right=tuple(box[3])
    
    # Next get min triangle, get those points.
    a, triangle = cv2.minEnclosingTriangle(points)
    return np.array([ (int(point[0]), int(point[1])) for point in triangle[:,0]])

    # Convert the points to tuple
    tri_bot_right=tuple(triangle[0][0])
    tri_top=tuple(triangle[1][0])
    tri_bot_left=tuple(triangle[2][0])
    
    rect_top=line(rect_top_left, rect_top_right)
    tri_left=line(tri_bot_left, tri_top)
    tri_right=line(tri_bot_right, tri_top)
    
    l_intersection=find_intersection(rect_top, tri_left)
    r_intersection=find_intersection(rect_top, tri_right)

    #returned in counter-clockwise from bottom-left
    #   botom_left
    #   botom_right
    #   top_right
    #   top_left
    if l_intersection and r_intersection:
        return np.array([
            (int(tri_bot_left[0]), int(tri_bot_left[1])),
            (int(tri_bot_right[0]), int(tri_bot_right[1])),
            (int(r_intersection[0]), int(r_intersection[1])),
            (int(l_intersection[0]), int(r_intersection[1]))
        ])
    else:
        return np.asarray([])

def get_homographic_image(img_src):

    pts_src = np.array(
    [
        [bot_left[0], bot_left[1]],
        [bot_right[0], bot_right[1]],
        [top_right[0], top_right[1]],
        [top_left[0], top_left[1]]    ]
    )
    pts_dst = np.array(
        [
            [0, 699],
            [399, 699],
            [399, 0],
            [0, 0]        
        ]
    )

    h, status = cv2.findHomography(pts_src, pts_dst)
    warped = cv2.warpPerspective(img_src, h, (400, 700))
    return warped

def find_puck(img_src):
    puckImage = img_src.copy()

    hsv_image=cv2.cvtColor(puckImage, cv2.COLOR_RGB2HSV)
    #hsv_mask=cv2.inRange(hsv_image,np.array([12,100,135]),np.array([30,150,200]))
    hsv_mask=cv2.inRange(hsv_image,np.array([20,100,100]),np.array([30,200,200]))

    opened=cv2.morphologyEx(hsv_mask,cv2.MORPH_OPEN, np.ones((9,9)))
    closed=cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((11,11)))

    edges = cv2.Canny(closed,255,255)
    x,y,w,h = cv2.boundingRect(edges)

    return x,y,w,h

def find_homograpy_points(img_src):
    wk_img = img_src.copy()

    bw_img = get_bw_img(wk_img)
    clean_img = remove_noise(bw_img)
    #res_img, contours, hierarchy = cv2.findContours(clean_img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #if (len(contours) < 10):
        #print 'There are too many contours to be accurate!', len(contours)


        #contours = None
        #max_try = 10
        #for i in range(0,max_try):
            #kernel_increase=60+i+1
            #kernel_o=np.zeroes((1,kernel_increase))
            #kernel_c=np.ones((50,60))

            #clean_img = remove_noise(bw_img, kernel_o, kernel_c)
            #res_img, contours, hierarchy = cv2.findContours(clean_img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            #if len(contours <= 10):
                #break

    #contour_points = get_contour_points(contours)
        
        # returned shape starts with bottom-left and 3 other points are
        # counter clockwise

    edges = cv2.Canny(clean_img,255,255)

    cv2.imshow('canny', edges)

    xs, ys = np.where(clean_img > 0)
    edgePts = np.array(zip(ys,xs))

    rect = cv2.minAreaRect(edgePts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(wk_img,[box],0,(0,0,255),2)

    # convert points to tuple
    rect_bot_left=tuple(box[0])
    rect_top_left=tuple(box[1])
    rect_top_right=tuple(box[2])
    rect_bot_right=tuple(box[3])

    a, triangle = cv2.minEnclosingTriangle(np.array([edgePts]))

    img = cv2.line(wk_img, (triangle[0][0][0],triangle[0][0][1]), (triangle[1][0][0], triangle[1][0][1]), (255,255,0), 5)
    img = cv2.line(wk_img, (triangle[1][0][0],triangle[1][0][1]), (triangle[2][0][0], triangle[2][0][1]), (255,255,0), 5)
    img = cv2.line(wk_img, (triangle[2][0][0],triangle[2][0][1]), (triangle[0][0][0], triangle[0][0][1]), (255,255,0), 5)

    

    # Convert the points to tuple
    tri_bot_right=tuple(triangle[0][0])
    tri_top=tuple(triangle[1][0])
    tri_bot_left=tuple(triangle[2][0])

    rect_top=line(rect_top_left, rect_top_right)
    tri_left=line(tri_bot_left, tri_top)
    tri_right=line(tri_bot_right, tri_top)

    l_intersection=find_intersection(rect_top, tri_left)
    r_intersection=find_intersection(rect_top, tri_right)

    print 'l_intersection:', l_intersection
    print 'r_intersection:', r_intersection

    board_shape = np.array([
        (int(tri_bot_left[0]), int(tri_bot_left[1])),
        (int(tri_bot_right[0]), int(tri_bot_right[1])),
        (int(r_intersection[0]), int(r_intersection[1])),
        (int(l_intersection[0]), int(r_intersection[1]))
    ])

    cv2.circle(wk_img, (board_shape[0][0],board_shape[0][1]), 40, (255, 0, 255), -1)
    cv2.circle(wk_img, (board_shape[1][0],board_shape[1][1]), 40, (255, 0, 255), -1)
    cv2.circle(wk_img, (board_shape[2][0],board_shape[2][1]), 40, (255, 0, 255), -1)
    cv2.circle(wk_img, (board_shape[3][0],board_shape[3][1]), 40, (255, 0, 255), -1)

    cv2.imshow('sample', wk_img)

    return board_shape
        


vs = WebcamVideoStream(src=1)
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

#r=redis.StrictRedis(host='localhost',port=6379,db=0)
while(True):
    img = vs.read()
    if img is not None:
        # cv2.imshow("Orig", img)
        try:     
            working_img = img.copy()
            
            #hsv_image=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            #corner_mask=getHSVMask(hsv_image.copy(),corner_lower,corner_upper)
            #corner_mask_open,corner_mask_close=getOpenCloseMask(corner_mask)
            #side_mask = getHSVMask(hsv_image.copy(),side_lower, side_upper)
            #side_mask_open, side_mask_close = getOpenCloseMask(side_mask)
            #goal_mask = getHSVMask(hsv_image.copy(),goal_lower, goal_upper)
            #goal_mask_open, goal_mask_close = getOpenCloseMask(goal_mask)
            
            #warped = get_homographic_image(img)
            #px,py,pw,ph = find_puck(warped)
            #puck_rect=cv2.rectangle(warped,(px,py),(px+pw,py+ph),(0,255,0),1)

            # centerW = pw/2
            # centerH = ph/2

            # centerx=px+centerW
            # centery=py+centerH

            # cv2.circle(warped, (centerx, centery), 8, (0, 255, 0), -1)
            # cv2.imshow("Homographic With Puck Detection", warped)

            # p=json.loads("{\"x\":0,\"y\":0}")
            # p['x'] = centerx
            # p['y'] = centery
            
            # r.set("machine-state-puck", json.dumps(p))
            # r.publish('state-changed', True)

            # todo: make this recursive?
            # If we have more than 15 contours, it's likely that our threshold is too low.

            #if contours is None:
                #contours = get_contours(working_img, 100)

            # if (len(contours) > 15):
            #      print "Too many contours..."
            #      contours = get_contours(img, 115)

            # contourPts = []
            # for i in range(0, len(contours)):
            #     for j in range(0, len(contours[i])):
            #         contourPts.append(contours[i][j])

            # contourPts = np.array(contourPts)

            # rect = cv2.minAreaRect(contourPts)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            #cv2.drawContours(img,[box],0,(255,0,0),2)

            h_points = np.asarray([])
            if frameCt < 100:
                h_points = find_homograpy_points(working_img)

            if frameCt % 200 == 0:
                h_points = find_homograpy_points(working_img)

            disp_img=working_img.copy()

            if not cur_h_points.any():
                cur_h_points = h_points
            elif h_points.any():
                if not np.max(h_points - cur_h_points) < 50:
                    cur_h_points = h_points
                else:
                    counter += 1
                    if counter > 100:
                        counter = 0
                        #cur_h_points = h_points
                        cur_h_points = np.mean( np.array([ cur_h_points, h_points ]), axis=0, dtype=np.int32)

            # if h_points is not False:
            #     cur_h_points = h_points
                

            if cur_h_points is not False:
                img = cv2.line(disp_img, tuple(cur_h_points[0]), tuple(cur_h_points[1]), (255,255,0), 5)
                img = cv2.line(disp_img, tuple(cur_h_points[1]), tuple(cur_h_points[2]), (255,255,0), 5)
                img = cv2.line(disp_img, tuple(cur_h_points[2]), tuple(cur_h_points[3]), (255,255,0), 5)
                img = cv2.line(disp_img, tuple(cur_h_points[3]), tuple(cur_h_points[0]), (255,255,0), 5)

            #cv2.drawContours(working_img,contours,-1,(255,0,0),2)

            # a, triangle = cv2.minEnclosingTriangle(contourPts)
            
            # img = cv2.line(img, (triangle[0][0][0],triangle[0][0][1]), (triangle[1][0][0], triangle[1][0][1]), (255,255,0), 2)
            # img = cv2.line(img, (triangle[1][0][0],triangle[1][0][1]), (triangle[2][0][0], triangle[2][0][1]), (255,255,0), 2)
            # img = cv2.line(img, (triangle[2][0][0],triangle[2][0][1]), (triangle[0][0][0], triangle[0][0][1]), (255,255,0), 2)

            # print 'Contours: ', len(contours)
            #cv2.imshow('edges', edges)

            font = cv2.FONT_HERSHEY_SIMPLEX
            ftext='Frame: ' + str(frameCt)
            cv2.putText(disp_img,ftext,(10,50), font, 0.5,(255,0,255),2,cv2.LINE_AA)

            fps_text='FPS: ' + str(cur_fps)
            cv2.putText(disp_img,fps_text,(10,100), font, 0.5,(0,255,0),2,cv2.LINE_AA)

            cv2.imshow("Preview", disp_img)

            fps.update()
            frames += 1
            frameCt += 1
        except Exception as ex:
            print ex
            pass
        
        if time.time() - start > 1:
            cur_fps = frames
            frames = 0
            start = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()