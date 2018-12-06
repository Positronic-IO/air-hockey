''' for each frame of out.mp4 that we've marked the puck in, 
    export the frame to a numbered jpg & export the class, 
    position, & size to a corresponding text file '''
import cv2
import numpy as np

pos = np.int0(np.load('labels.npy'))
vs = cv2.VideoCapture('data.mp4')
puck_size = 10
frame_index = -1
while True:
	_, frame = vs.read()
	if frame is None:
		break
	frame = cv2.resize(frame, (240, 320) , interpolation = cv2.INTER_LINEAR)
	#start_y, start_x = 120, 65
	#frame = frame[start_y:-140,start_x:-170]
	frame_index += 1
	if np.sum(pos[frame_index]) > 0:
		# save the data
		image_name = "{}.jpg".format(str(frame_index))
		cv2.imwrite(image_name, frame)

		# save the labels
		x_pixels, y_pixels = pos[frame_index][0], pos[frame_index][1]
		width, height = puck_size / frame.shape[1], puck_size / frame.shape[0]
		x, y = x_pixels /2 / frame.shape[1], y_pixels / 2 / frame.shape[0]
		text_name = "{}.txt".format(str(frame_index))
		with open(text_name, "w") as text_file:
			print("0 {} {} {} {}".format(x, y, width, height), file=text_file)

vs.release()
