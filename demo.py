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

def annotate(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ## isolate HUE
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,1:] = 0
    t, hue = cv2.threshold(hsv, 50, 255, cv2.THRESH_BINARY)
    hue = ~hue[:,:,0]

    ## denoise
    kernel = np.ones((5, 50))
    opened = cv2.morphologyEx(hue, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((10,10)))

    ## edges are less work
    laplacian = cv2.Laplacian(closed,cv2.CV_8UC1)

    ## find largest external blob
    im2, outer_contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(outer_contours) == 0:
        return rgb
    max_contour = max(outer_contours, key = cv2.contourArea)

    ## translate playing area to a clean rect
    rect = cv2.minAreaRect(max_contour)
    box = np.int0(cv2.boxPoints(rect))

    ## get all of the shapes
    im2, all_contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ## isolate the shapes inside the table area
    def point_in_box(point, box):
        retval = point[0] > min(box[:,0]) and point[0] < max(box[:,0]) and point[1] > min(box[:,1]) and point[1] < max(box[:,1])
        return retval

    contours_in_box = [c for c in all_contours if np.all(point_in_box(c[0,0,:], box))]

    ## filter out large shapes
    def get_area(c):
        m = cv2.moments(c)
        return m['m00']
    max_area = get_area(max_contour)
    small_contours = [c for c in contours_in_box if (get_area(c) / max_area) < 0.5]

    ## filter out non-circular shapes
    circular_contours = [c for c in small_contours if len(cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)) < 15]

    ## put it all together
    cv2.drawContours(rgb, [box], 0, (255, 0, 0), 3)
    cv2.drawContours(rgb, circular_contours, -1, (0,255,0), 5)
    return rgb

def main():
    vs = WebcamVideoStream(src=0)
    vs = vs.start()

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        img = imutils.resize(vs.read(), width=1024)
        annotated = annotate(img)
        cv2.imshow("test", annotated)

    cv2.destroyAllWindows()
    vs.stop()

main()