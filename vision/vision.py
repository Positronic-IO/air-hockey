import numpy as np
import cv2
import imutils
from WebcamVideoSteam import WebcamVideoStream
from imutils.video import FPS
import redis
import json
from robotvision import RobotVision as rv

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

def find_trapezoid(img_src):
    wk_img = img_src.copy()

    bw_img = rv.get_bw_img(wk_img)
    clean_img = rv.remove_noise(bw_img)
    res_img, contours, hierarchy = cv2.findContours(clean_img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) > 10):
        print 'There are too many contours to be accurate!', len(contours)


        contours = None
        max_try = 10
        for i in range(0,max_try):
            kernel_increase=60+i+1
            clean_img = rv.remove_noise(bw_img, kernel_open=np.zeroes(1,kernel_increase))
            res_img, contours, hierarchy = cv2.findContours(clean_img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            if len(contours <= 10):
                break

    contour_points = rv.get_contour_points(contours)
    
    # returned shape starts with bottom-left and 3 other points are
    # counter clockwise
    board_shape=get_board_shape(contour_points)

    return board_shape
        


vs = WebcamVideoStream(src=1)
vs = vs.start()
fps = FPS().start()
import time
start = time.time()
frames = 0
frameCt = 0

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

            trap = find_trapezoid(working_img)
            
            if frameCt % 100 == 0:
                contours = get_contours(vs.read().copy(), 100)

            cv2.drawContours(working_img,contours,-1,(255,0,0),2)

            # a, triangle = cv2.minEnclosingTriangle(contourPts)
            
            # img = cv2.line(img, (triangle[0][0][0],triangle[0][0][1]), (triangle[1][0][0], triangle[1][0][1]), (255,255,0), 2)
            # img = cv2.line(img, (triangle[1][0][0],triangle[1][0][1]), (triangle[2][0][0], triangle[2][0][1]), (255,255,0), 2)
            # img = cv2.line(img, (triangle[2][0][0],triangle[2][0][1]), (triangle[0][0][0], triangle[0][0][1]), (255,255,0), 2)
            

            cv2.imshow("Preview", working_img)

            # print 'Contours: ', len(contours)
            #cv2.imshow('edges', edges)

            fps.update()
            frames += 1
        except Exception as ex:
            print ex
            pass
        
        if time.time() - start > 1:
            print "FPS: ", frames
            frames = 0
            start = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()