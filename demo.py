import numpy as np
import cv2
import imutils
import skimage

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

import cv2
import imutils
from imutils.video import FPS, WebcamVideoStream
from imutils import perspective

boxes = []

def annotate(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ## isolate HUE
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,1:] = 0
    t, hue = cv2.threshold(hsv, 50, 255, cv2.THRESH_BINARY)
    hue = ~hue[:,:,0]

    ## mask out black
    t, blue = cv2.threshold(rgb, 20, 255, cv2.THRESH_BINARY_INV, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    mask = ~blue[:,:,2]
    masked = mask & hue
    
    ## denoise
    kernel = np.ones((1, 1))
    opened = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((1,5)))

    ## edges are less work
    laplacian = cv2.Laplacian(closed,cv2.CV_8UC1)

    ## find largest external blob
    im2, outer_contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(outer_contours, key = cv2.contourArea)

    ## translate playing area to a clean rect
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    ordered = perspective.order_points(box)
    boxes.append(ordered)
    if len(boxes) > 50:
        boxes.pop(0)
    table = np.int0(np.mean(boxes, axis=0))

    ## get all of the shapes
    im2, all_contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ## isolate the shapes inside the table area
    contours_in_box = [c for c in all_contours if not (min(c[:,0,0]) <= min(table[:,0]) or max(c[:,0,0]) >= max(table[:,0]) or min(c[:,0,1]) <= min(table[:,1]) or max(c[:,0,1]) >= max(table[:,1]))]

    ## filter out large shapes
    def get_area(c):
        m = cv2.moments(c)
        return m['m00']
    max_area = get_area(max_contour)
    small_contours = [c for c in contours_in_box if (get_area(c) * 100 / max_area) < 50 and (get_area(c) * 100 / max_area) > 0.1]

    ## filter out non-circular shapes
    def is_circle(c):
        polys = len(cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True))
        if polys < 8 or polys > 15:
            return False

        center, radius = cv2.minEnclosingCircle(c)
        area = get_area(c)
        circle_area = 3.14*radius*radius
        if abs(area-circle_area) > .5 * area:
            return False

        return True

    circular_contours = [c for c in small_contours if is_circle(c)]

    ## put it all together
    cv2.drawContours(img, [table], 0, (255, 0, 0), 2)
    for c in circular_contours:
        center, radius = cv2.minEnclosingCircle(c)
        cv2.circle(img, (int(center[0]), int(center[1])), int(radius), (0,255,0), 2)
#    cv2.drawContours(img, circular_contours, -1, (0,255,0), 2)
    return img
    
def main():
    vs = WebcamVideoStream(src=0)
    vs = vs.start()

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        img = vs.read()
        #img = imutils.resize(img, width=1024)
        annotated = annotate(img)
        cv2.imshow("test", annotated)

    cv2.destroyAllWindows()
    vs.stop()

main()