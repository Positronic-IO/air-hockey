import numpy as np
import cv2

class RobotVision:

    # Clean noise from mask
    @staticmethod
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

    # Get HSV Mask from image based on ranges
    @staticmethod
    def getHSVMask(hsv_img, range_lower_bound, range_upper_bound):
        hsv_mask=cv2.inRange(hsv_img,range_lower_bound,range_upper_bound)
        return hsv_mask

    # Get clearer mask
    @staticmethod
    def getOpenCloseMask(orig_mask):
        kernelOpen=np.ones((5,5))
        kernelClose=np.ones((20,20))
        
        mask_open=cv2.morphologyEx(orig_mask,cv2.MORPH_OPEN,kernelOpen)
        mask_close=cv2.morphologyEx(orig_mask,cv2.MORPH_CLOSE,kernelClose)
        
        return mask_open,mask_close

    # Get rectangle of item based on its mask.
    @staticmethod
    def getMaskRectangle(clean_mask):
        # Get contours
        _, contours, _=cv2.findContours(clean_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        # Determine biggest contour and use that.
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        #print 'biggest_contour';
        #biggest_contour;
        
        # Generate rect from contour
        x,y,w,h = cv2.boundingRect(biggest_contour)
        
        return x,y,w,h