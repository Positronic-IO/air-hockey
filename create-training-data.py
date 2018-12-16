from __future__ import print_function
from imutils.video import WebcamVideoStream
import cv2
from timeit import default_timer as timer
import numpy as np
import datetime
import os

window_name = "science!"
slider_name = 'Frame'
cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

vs = cv2.VideoCapture('out.mp4')
frame_count = int(cv2.VideoCapture.get(vs, int(cv2.CAP_PROP_FRAME_COUNT)))

def process_frame(frame):
	global fps, out, frame_index
	out = frame

	if np.sum(pos1[frame_index]) > 0:
		drawX(pos1[frame_index][0], pos1[frame_index][1], left_color)
	if np.sum(pos2[frame_index]) > 0:
		drawX(pos2[frame_index][0], pos2[frame_index][1], right_color)
	
	fps_out = "{}/{}".format(str(frame_index), str(frame_count))
	cv2.putText(out, fps_out, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,0, 0), 2)

	cv2.imshow(window_name, out)

def on_trackbar(pos, data = None):
	global frame_index
	cv2.VideoCapture.set(vs, int(cv2.CAP_PROP_POS_FRAMES), pos)
	_, frame = vs.read()
	frame_index = pos
	process_frame(frame)

cv2.createTrackbar(slider_name, window_name, 1, frame_count, on_trackbar)
frame_index = -1

def init_pos(index):
	path = 'pos-{}.npy'.format(str(index))
	if os.path.exists(path):
		ret = np.int0(np.load(path))
	else:
		ret = np.zeros((frame_count,2))
	return ret
pos1 = init_pos(1)
pos2 = init_pos(2)
left_color = (26, 253, 141)
right_color = (255, 255, 255)

def drawX(x, y, color):
	global out, window_name
	text = "X"
	font_face = cv2.FONT_HERSHEY_PLAIN
	font_scale = 2
	thickness = 2
	size = cv2.getTextSize(text, font_face, font_scale, thickness)
	cv2.putText(out, text, (x - int(size[0][0] / 2), y + int(size[0][1] / 2)), font_face, font_scale, color, thickness)
	print(x,y)

def callback(event, x, y, flags, param):
	if event == cv2.EVENT_RBUTTONDOWN:
		pos2[frame_index] = np.int0(np.asarray([x,y]))
		drawX(x, y, right_color)
		cv2.imshow(window_name, frame)

	if event == cv2.EVENT_LBUTTONDOWN:
		if flags & cv2.EVENT_FLAG_CTRLKEY:
			pos2[frame_index] = np.int0(np.asarray([x,y]))
			drawX(x, y, right_color)
			cv2.imshow(window_name, frame)
		else:
			pos1[frame_index] = np.int0(np.asarray([x,y]))
			drawX(x, y, left_color) 
			cv2.imshow(window_name, frame)
cv2.setMouseCallback(window_name, callback)

while(True):
	_, frame = vs.read()
	frame_index += 1
	cv2.setTrackbarPos(slider_name, window_name, frame_index) 

	process_frame(frame)
	
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

np.save("pos-1", pos1)
np.save("pos-2", pos2)

cv2.destroyAllWindows()
vs.release()
