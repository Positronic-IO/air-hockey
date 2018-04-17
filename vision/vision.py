import numpy as np
import cv2
from robotvision.robotvision import RobotVision

#Variables...

# Puck HSV Bounds (Green)
pucklowerBound=np.array([30,150,90])
puckupperBound=np.array([90,190,140])

# Bot HSV Bounds (White 3dPrint)
botlowerBound=np.array([95,15,165])
botupperBound=np.array([110,30,180])

'''

For reference, the first argument in the cap.set() command refers to the enumeration of the camera properties, listed below:

0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
5. CV_CAP_PROP_FPS Frame rate.
6. CV_CAP_PROP_FOURCC 4-character code of codec.
7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

'''

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,340)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,220)
cap.set(cv2.CAP_PROP_FPS,120)

while(True):

    
    ret, img = cap.read()
    cv2.imshow('img-vid', img)

    working_img=cv2.resize(img,(340,220))

    #convert image to HSV
    hsv_image=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # DEGUG: Show hsv image
    #cv2.imshow('hsv-vid', hsv_image)

    # Get HSV Mask for bot and puck from HSV image.
    puck_mask=RobotVision.getHSVMask(hsv_image,pucklowerBound, puckupperBound)
    bot_mask=RobotVision.getHSVMask(hsv_image,botlowerBound,botupperBound)

    # DEGUG: Show mask
    #cv2.imshow('puck-mask', puck_mask)
    #cv2.imshow('bot-mask', bot_mask)

    #Get open & Close mask
    puck_mask_open,puck_mask_close= RobotVision.getOpenCloseMask(puck_mask)
    bot_mask_open,bot_mask_close= RobotVision.getOpenCloseMask(bot_mask)

    #cv2.imshow(puck_mask_close)
    #cv2.imshow(bot_mask_close)

    # Clean Noise form mask
    #puck_mask_close_cleaned = RobotVision.cleanMask(puck_mask_close)
    #bot_mask_close_cleaned = RobotVision.cleanMask(bot_mask_close)

    #cv2.imshow('clean-puck-mask', puck_mask_close)
    #cv2.imshow(bot_mask_close_cleaned)

    try:
        px,py,pw,ph = RobotVision.getMaskRectangle(puck_mask_close)
        bx,by,bw,bh = RobotVision.getMaskRectangle(bot_mask_close)

        # Draw locations of puck and bot on image.
        image = img.copy()
        cv2.rectangle(image,(px,py),(px+pw,py+ph),(255,0,0),2)
        cv2.rectangle(image,(bx,by),(bx+bw,by+bh),(0,0,255),1)

        cv2.imshow('', image)
    except ValueError as identifier:
        cv2.imshow('', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()