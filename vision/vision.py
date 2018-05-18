import numpy as np
import cv2
import imutils
from WebcamVideoSteam import WebcamVideoStream
from imutils.video import FPS
#import redis
import json


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
    
    # Apply threshold to grayscale img. (Converts to b&w)
    t, bw = cv2.threshold(wk_img, threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Global Threshold', bw)
    
    return bw.copy()

def remove_noise(image):
    wk_img=image.copy()
    
    # Baseline params
    kernel1=5
    kernel2=60
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

    # this cleans up the noise on the inside and reduces the number of
    # objects/contours.
    closed=cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((10,10)))
    
    return closed.copy()

def find_homograpy_points(img_src):
    #todo: Split this guy out into functions
    wk_img = img_src.copy()

    gray = cv2.cvtColor(wk_img, cv2.COLOR_BGR2GRAY)
    bw_img = get_bw_img(image=gray, threshold=140)

    clean_img = remove_noise(bw_img)

    zero_count=np.count_nonzero(clean_img)
    #print "Zero-Count:", zero_count

    edges = cv2.Canny(clean_img,1,1)

    xs, ys = np.where(edges > 0)
    edgePts=np.array(zip(ys,xs))

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

    homography_points = np.array([
        (int(tri_bot_left[0]), int(tri_bot_left[1])),
        (int(tri_bot_right[0]), int(tri_bot_right[1])),
        (int(r_intersection[0]), int(r_intersection[1])),
        (int(l_intersection[0]), int(l_intersection[1]))
    ])

    # Preview
    # disp=working_img.copy()
    # cv2.drawContours(disp,[box],0,(255,0,255),2)
    # img = cv2.line(disp, tri_bot_left, tri_top, (255,255,0), 2)
    # img = cv2.line(disp, tri_top, tri_bot_right, (255,255,0), 2)
    # img = cv2.line(disp, tri_bot_right, tri_bot_left, (255,255,0), 2)
    # img = cv2.polylines(disp, [homography_points], True, (255, 0, 0), 2)
    # cv2.imshow('edges', edges)
    # cv2.imshow('Mapping', disp)

    return homography_points
        
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
    warped = cv2.warpPerspective(wk_img, h, (480, 640))

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
    xs, ys = np.where(closed > 0)
    pts = np.array(zip(ys, xs))

    if (len(pts) > 0):
        center, radius = cv2.minEnclosingCircle(pts)

        return tuple(np.int0(center)), int(radius)
    else:
        print 'Puck Not Detected'
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
    xs, ys = np.where(closed > 0)
    pts = np.array(zip(ys, xs))

    if (len(pts) > 0):
        center, radius = cv2.minEnclosingCircle(pts)

        return tuple(np.int0(center)), int(radius)
    else:
        print 'Bot Not Detected'
        return (0,0), 0

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

            h_points = np.asarray([])
            if frameCt < 100:
                h_points = find_homograpy_points(working_img)

            if frameCt % 800 == 0:
                h_points = find_homograpy_points(working_img)

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

            h_img = get_homographic_image(working_img, cur_h_points)            


            puck_center, puck_radius = find_puck(h_img)
            bot_center, bot_radius = find_bot(h_img)

            print "Puck Center:", puck_center
            print "Bot Center:", bot_center

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