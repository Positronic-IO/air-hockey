# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2
from timeit import default_timer as timer
import numpy as np

import skimage

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

from imutils.video import FPS, WebcamVideoStream
from imutils import perspective

# import the necessary packages
import datetime

class piece:
    def __init__(self, circle):
        self.speed = 0
        self.retries = 0
        self.direction = 0
        self.update(circle)

    def update(self, circle):
        self.center = (int(circle[0][0]), int(circle[0][1]))
        self.radius = int(circle[1])
    
    def dist(self, circle):
        return np.sum(abs(np.asarray(self.center) - np.asarray(circle[0])))

    def draw(self, out):
        cv2.circle(out, self.center, self.radius, (0,255,0), 2)

class cheap_fps:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self.fps_buffer = []
		self.last = timer()

	def update(self):
		now = timer()
		diff = now - self.last
		self.last = now
		fps = 1 / diff
		self.fps_buffer.append(fps)
		if len(self.fps_buffer) > 10:
			self.fps_buffer.pop(0)
		return str(int(np.mean(self.fps_buffer)))

boxes = []
pieces = []
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
    
    out = np.zeros_like(img)
#    out = img

    ## put it all together
    #cv2.drawContours(out, [table], 0, (255, 0, 0), 2) # draw table
    circles = []
    for c in circular_contours:
        center, radius = cv2.minEnclosingCircle(c)
        if radius < 20 and radius > 10:
            circles.append( (center, radius) )

    if len(circles) < len(pieces):
        # remove from pieces
        extras = list(pieces)
        for circle in circles:
            diff = [ extra.dist(circle) for extra in extras ]
            index = diff.index(np.min(diff))
            pieces[pieces.index(extras[index])].retries = 0
            extras.pop(index)
        for extra in extras:
            extra.retries += 1
            if extra.retries > 2:
                pieces.pop(pieces.index(extra))

    if len(circles) > len(pieces):
        # add to pieces
        for item in pieces:
            diff = [ item.dist(circle) for circle in circles ]
            index = diff.index(np.min(diff))
            circles.pop(index)
        for circle in circles:
            pieces.append(piece(circle))

    # do something
    for circle in circles:
        diff = [ item.dist(circle) for item in pieces ]
        index = diff.index(np.min(diff))
        pieces[index].update(circle)

    for item in pieces:
        item.draw(out)

#    cv2.drawContours(img, circular_contours, -1, (0,255,0), 2)

    # crop to detected table
#    out = out[min(table[:,1]): max(table[:,1]), min(table[:,0]): max(table[:,0])]

    return out


vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
cfps = cheap_fps()

cv2.namedWindow('science!', cv2.WINDOW_NORMAL)
cv2.setWindowProperty("science!",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# loop over some frames...this time using the threaded stream
while(True):
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	annotated = annotate(frame)
	cropped = annotated[150:-190,130:-220]
#	resized = imutils.resize(annotated, width=1280, height=2280)
	out = cropped
    
	# update the FPS counter
	fps.update()
	fps_out = cfps.update()
	cv2.putText(out, fps_out, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, (255,0, 0), 2)

	# check to see if the frame should be displayed to our screen
	cv2.imshow("science!", out)

# stop the timer and display FPS information
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()