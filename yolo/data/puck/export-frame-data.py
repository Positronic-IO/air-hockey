''' for each frame of out.mp4 that we've marked the puck in, 
    export the frame to a numbered jpg & export the class, 
    position, & size to a corresponding text file '''
import cv2
import numpy as np

pos = np.int0(np.load('labels.npy'))
vs = cv2.VideoCapture('data.mp4')

frame_index = -1
while True:
	_, frame = vs.read()
	if frame is None:
		break
	frame_index += 1
	if frame is not None and np.sum(pos[frame_index]) > 0:
		# save the data
		image_name = "{}.jpg".format(str(frame_index))
		cv2.imwrite(image_name, frame)

		# save the labels
		width = 20 / frame.shape[1]
		height = 20 / frame.shape[0]
		x = pos[frame_index][0] / frame.shape[1]
		y = pos[frame_index][1] / frame.shape[0]
		text_name = "{}.txt".format(str(frame_index))
		with open(text_name, "w") as text_file:
			print("0 {} {} {} {}".format(x, y, width, height), file=text_file)

vs.release()
