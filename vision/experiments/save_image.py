import numpy as np
import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,340)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,220)
cap.set(cv2.CAP_PROP_FPS,120)

out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()