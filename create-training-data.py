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
	t, black = cv2.threshold(rgb, 20, 255, cv2.THRESH_BINARY_INV, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
	mask = ~black[:,:,2]
	masked = mask & hue

	# isolate blue	
	hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
	blue_lower=np.array([80,0,0],np.uint8)
	blue_upper=np.array([255,255,255],np.uint8)
	blue_mask = cv2.inRange(hsv, blue_lower, blue_upper) 	
	blue = cv2.bitwise_and(rgb, rgb, mask=blue_mask)
	masked = blue[:,:,1] & mask

	## make it fuzzy
	eroded = cv2.erode(masked, None, iterations=2)
	fuzzed = cv2.dilate(eroded, None, iterations=5)

	## edges are less work
	laplacian = cv2.Laplacian(fuzzed,cv2.CV_8UC1)

	## find largest external blob
	im2, outer_contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	max_contour = max(outer_contours, key = cv2.contourArea)
	blob = cv2.approxPolyDP(max_contour, 0.01*cv2.arcLength(max_contour, True), True)

	## translate playing area to a clean rect
	rect = cv2.minAreaRect(blob)
	box = cv2.boxPoints(rect)
	ordered = perspective.order_points(box)
	boxes.append(ordered)
	if len(boxes) > 50:
		boxes.pop(0)
	table = np.int0(np.mean(boxes, axis=0))

	## get not blue inside the box
	not_blue = ~masked
	table_mask = np.zeros_like(masked)
	cv2.fillPoly(table_mask, np.array([box], dtype=np.int32), 255)
	not_blue = not_blue & table_mask
	t, thresh = cv2.threshold(not_blue, 254, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

	## edges are less work
	laplacian = cv2.Laplacian(thresh, cv2.CV_8UC1)

	## get all of the shapes
	im2, all_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)                                                   
	test = cv2.drawContours(np.zeros_like(laplacian), all_contours, -1, (255,255,255), 1)

	## get all of the shapes
	im2, all_contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	## filter by size
	def get_area(c):
		m = cv2.moments(c)
		return m['m00']
	small_contours = [c for c in all_contours if get_area(c) < 400 and get_area(c) > 50]

	## filter out non-circular shapes
	def is_circle(c):
		center, radius = cv2.minEnclosingCircle(c)
		area = get_area(c)
		circle_area = 3.14*radius*radius
		if abs(area-circle_area) > .5 * area:
			return False

		return True

	circular_contours = [c for c in small_contours if is_circle(c)]
	
	out = np.zeros_like(img)
	out = img

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
	#out = out[min(table[:,1]): max(table[:,1]), min(table[:,0]): max(table[:,0])]
	coords = (min(table[:,1]), max(table[:,1]), min(table[:,0]), max(table[:,0]))
	return (out, coords)


vs = cv2.VideoCapture('out.mp4')

frame_count = int(cv2.VideoCapture.get(vs, int(cv2.CAP_PROP_FRAME_COUNT)))
frame_index = -1
#vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
cfps = cheap_fps()

import os
if os.path.exists('pos.npy'):
	pos = np.int0(np.load('pos.npy'))
else:
	pos = np.zeros((frame_count,2))

def drawX(x, y):
	global out, window_name
	text = "X"
	font_face = cv2.FONT_HERSHEY_PLAIN
	font_scale = 2
	thickness = 2
	size = cv2.getTextSize(text, font_face, font_scale, thickness)
	cv2.putText(out, text, (x - int(size[0][0] / 2), y + int(size[0][1] / 2)), font_face, font_scale, (255,0, 0), thickness)
	print(x,y)

def callback(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		pos[frame_index] = np.int0(np.asarray([x,y]))
		drawX(x, y)
		cv2.imshow(window_name, frame)

window_name = "science!"
#cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, callback)

# loop over some frames...this time using the threaded stream
while(True):
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	_, frame = vs.read()
	# frame = vs.read()
	frame_index += 1
#	frame = cv2.imread(r'notebooks/overhead-science-4.png')

#	frame = imutils.resize(frame, width=800)
#	annotated, coords = annotate(frame)
#	cropped = annotated[coords[0]:coords[1], coords[2]:coords[3]]
	#trim = int(np.asarray(annotated).shape[1]*0.1)
	#cropped = annotated[:, trim:-1*trim]

#	resized = imutils.resize(annotated, width=1280, height=2280)
	out = frame
	if np.sum(pos[frame_index]) > 0:
		drawX(pos[frame_index][0], pos[frame_index][1])
	
	# update the FPS counter
	fps.update()
	fps_out = cfps.update()
	cv2.putText(out, fps_out, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,0, 0), 2)

	# check to see if the frame should be displayed to our screen
	cv2.imshow(window_name, out)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

np.save("pos", pos)

# stop the timer and display FPS information
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
#vs.stop()