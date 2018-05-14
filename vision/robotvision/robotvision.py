import numpy as np
from __future__ import division
import cv2

class RobotVision:

    #Return a line based on two points.
    @staticmethod
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    # find intersection between two lines.
    @staticmethod
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

    @staticmethod
    def get_bw_img(image, threshold=150):
        wk_img = image.copy()
        
        # Convert image to grayscale
        gray = cv2.cvtColor(wk_img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to grayscale img. (Converts to b&w)
        t, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        return bw.copy()

    @staticmethod
    def remove_noise(image, kernel_open=np.ones((1,60)), kernel_close=np.ones((50,60))):
        wk_img=image.copy()
        
        # we only want large objects since we are trying to detect
        # the board. 1x60 is a good starting point. Can call again
        # with override if the result is not good enough.
        # this cleans up the outside noise
        opened=cv2.morphologyEx(wk_img,cv2.MORPH_OPEN, kernel_open)
        
        # this cleans up the noise on the inside and reduces the number of
        # objects.
        closed=cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        return closed.copy()

    @staticmethod
    def get_contour_points(contours):
        pts = []
        for i in range(0, len(contours)):
            for j in range(0, len(contours[i])):
                pts.append(contours[i][j])

        return np.array(pts)

    @staticmethod
    def get_board_shape(points):

        # first, get minarearect and get box points from the rect.
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # convert points to tuple
        rect_bot_left=tuple(box[0])
        rect_top_left=tuple(box[1])
        rect_top_right=tuple(box[2])
        rect_bot_right=tuple(box[3])
        
        # Next get min triangle, get those points.
        a, triangle = cv2.minEnclosingTriangle(points)
        
        # Convert the points to tuple
        tri_bot_right=tuple(triangle[0][0])
        tri_top=tuple(triangle[1][0])
        tri_bot_left=tuple(triangle[2][0])
        
        rect_top=line(rect_top_left, rect_top_right)
        tri_left=line(tri_bot_left, tri_top)
        tri_right=line(tri_bot_right, tri_top)
        
        l_intersection=intersection(rect_top, tri_left)
        r_intersection=intersection(rect_top, tri_right)

        #returned in counter-clockwise from bottom-left
        #   botom_left
        #   botom_right
        #   top_right
        #   top_left
        if l_intersection and r_intersection:
            return np.array([
                (int(tri_bot_left[0]), int(tri_bot_left[1])),
                (int(tri_bot_right[0]), int(tri_bot_right[1])),
                (int(r_intersection[0]), int(r_intersection[1])),
                (int(l_intersection[0]), int(r_intersection[1]))
            ])
        else:
            return False