import numpy as np
import cv2
import imutils
from WebcamVideoSteam import WebcamVideoStream
from imutils.video import FPS

corner_lower=np.array([75,106,120])
corner_upper=np.array([90,210,200])

side_lower = np.array([47,47,110])
side_upper = np.array([62,112,174])

goal_lower = np.array([17,114,87])
goal_upper = np.array([26,168,111])

def cleanMask(dirty_mask):
    points = []
    for i in range(dirty_mask.shape[0]):
        for j in range(dirty_mask.shape[1]):
            if dirty_mask[i, j] == 255:
                points.append((i,j))
    #Remove exterior dots
    padding = 2
    from collections import deque
    remaining_points = set(points)
    saved_locations = []
    while remaining_points:
        current_point = remaining_points.pop()
        y = current_point[0]
        x = current_point[1]
        saved_points = set()
        horizon = deque([(y-1,x-1),(y+1,x-1),(y-1,x+1),(y+1,x+1)])
        saved_points.add(current_point)
        while(len(horizon) > 0):
            to_check = horizon.pop()
            y = to_check[0]
            x = to_check[1]
            if to_check in remaining_points:
                remaining_points.discard(to_check)
                horizon.appendleft((y-1,x-1))
                horizon.appendleft((y+1,x-1))
                horizon.appendleft((y-1,x+1))
                horizon.appendleft((y+1,x+1))
                saved_points.add(to_check)
        saved_locations.append(saved_points)
    cleaned_mask = dirty_mask.copy()
    cleaned_mask[:,:] = 0
    max_location = set()
    for location in saved_locations:
        if location:
            if len(location) > len(max_location):
                max_location = location

    for y, x in max_location:
        cleaned_mask[y-padding:y+padding,x-padding:x+padding] = 255
        
    return cleaned_mask

def getHSVMask(hsv_img, range_lower_bound, range_upper_bound):
    hsv_mask=cv2.inRange(hsv_img,range_lower_bound,range_upper_bound)
    return hsv_mask

def getOpenCloseMask(orig_mask):
    kernelOpen=np.ones((3,3))
    kernelClose=np.ones((10,10))
    
    mask_open=cv2.morphologyEx(orig_mask,cv2.MORPH_OPEN,kernelOpen)
    mask_close=cv2.morphologyEx(orig_mask,cv2.MORPH_CLOSE,kernelClose)
    
    return mask_open,mask_close

def getMaskRectangle(clean_mask, offset=0):
    # Get contours
    _, contours, _=cv2.findContours(clean_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    # Determine biggest contour and use that.
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    contour_sizes.sort(key= lambda x: x[0], reverse=True)
    #print(contour_sizes)
    biggest_contour = contour_sizes[offset][1]

    #print 'biggest_contour';
    #biggest_contour;
    
    # Generate rect from contour
    x,y,w,h = cv2.boundingRect(biggest_contour)
    
    return x,y,w,h

def get_rect_mid(x, y, w, h):
    return x + w / 2, y + h / 2

def get_rect_br(x, y, w, h):
    return x + w, y + h

def get_rect_bl(x, y, w, h):
    return x, y + h

def get_coordinates(corner_mask, side_mask, goal_mask):
    rects = [getMaskRectangle(corner_mask), getMaskRectangle(corner_mask, 1), getMaskRectangle(corner_mask, 2), getMaskRectangle(corner_mask, 3)]
    rects.sort()
    #print(rects)
    coords = {}
    to_add = get_rect_bl(*rects[0])
    if to_add[0] < 640 and to_add[1] > 360:
        coords[(0,0)]     = to_add
    to_add = get_rect_mid(*rects[1])
    if to_add[0] < 640 and to_add[1] < 360:
        coords[(0,699)]   = to_add
    to_add = get_rect_mid(*rects[2])
    if to_add[0] > 640 and to_add[1] < 360:
        coords[(399,699)]   = to_add
    to_add = get_rect_br(*rects[3])
    if to_add[0] > 640 and to_add[1] > 360:
        coords[(399,0)]   = to_add
    side_rects = [getMaskRectangle(side_mask), getMaskRectangle(side_mask, 1)]
    side_rects.sort()
    if len(side_rects) == 2:
        coords[(-50, 400)]  = get_rect_mid(*side_rects[0])
        coords[(449, 400)]  = get_rect_mid(*side_rects[1])
    coords[(200, 729)] = get_rect_mid(*getMaskRectangle(goal_mask))
    #print(coords)
    return coords

def get_homographic_image(img_src, c_mask, s_mask, g_mask):
    coords = get_coordinates(c_mask, s_mask, g_mask)
    keys, values = zip(*coords.items())
    pts_dst = np.array(keys)
    pts_src = np.array(values)
    h, status = cv2.findHomography(pts_src, pts_dst)
    warped = cv2.warpPerspective(img_src, h, (400, 700))
    return warped

vs = WebcamVideoStream(src=0)
vs = vs.start()
fps = FPS().start()
import time
start = time.time()
frames = 0
while(True):
    img = vs.read()
    if img is not None:
        #cv2.imshow("Orig", img)
        try:
            
            
            hsv_image=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            corner_mask=getHSVMask(hsv_image.copy(),corner_lower,corner_upper)
            corner_mask_open,corner_mask_close=getOpenCloseMask(corner_mask)
            side_mask = getHSVMask(hsv_image.copy(),side_lower, side_upper)
            side_mask_open, side_mask_close = getOpenCloseMask(side_mask)
            goal_mask = getHSVMask(hsv_image.copy(),goal_lower, goal_upper)
            goal_mask_open, goal_mask_close = getOpenCloseMask(goal_mask)
            
            warped = get_homographic_image(img, corner_mask_open, side_mask_open, goal_mask_open)
            #cv2.imshow("Homographic", warped)
            fps.update()
            frames += 1
        except Exception as ex:
            pass
        
        if time.time() - start > 1:
            print(frames)
            frames = 0
            start = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()