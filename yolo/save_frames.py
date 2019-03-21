""" Save frames of videos """
import cv2


if __name__ == "__main__":

    vidcap = cv2.VideoCapture('yolo/data/data.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("yolo/captured_frames/frame%d.jpg" %
                    count, image)     # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
