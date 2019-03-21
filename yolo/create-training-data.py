import cv2
import numpy as np
import os


class markers:
    def __init__(self, frame_count):
        self.pos_green = self.load(1, frame_count)
        self.pos_white = self.load(2, frame_count)
        self.color_green = (26, 253, 141)
        self.color_white = (255, 255, 255)

    def load(self, index, frame_count):
        path = 'pos-{}.npy'.format(str(index))
        return np.int0(np.load(path)) if os.path.exists(path) else np.zeros((frame_count, 2))

    def release(self):
        np.save("pos-1", self.pos_green)
        np.save("pos-2", self.pos_white)

    def draw(self, index, frame):
        draw_these = []
        if len(self.pos_green) > 0 and np.sum(self.pos_green[index]) > 0:
            draw_these.append((self.pos_green[index], self.color_green))
        if len(self.pos_white) > 0 and np.sum(self.pos_white[index]) > 0:
            draw_these.append((self.pos_white[index], self.color_white))

        [self.drawX(frame, m[0][0], m[0][1], m[1]) for m in draw_these]

    def drawX(self, out, x, y, color):
        size = cv2.getTextSize(
            text="X", fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=2)
        cv2.putText(img=out, text="X", org=(x - int(size[0][0] / 2), y + int(size[0][1] / 2)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=color, thickness=2)


class ui:
    def __init__(self, video_path):
        self.window_name = "science!"
        self.slider_name = 'Frame'
        self.frame_index = 0
        self.frame_current = None

        self.vs = cv2.VideoCapture(video_path)
        frame_count = int(cv2.VideoCapture.get(
            self.vs, int(cv2.CAP_PROP_FRAME_COUNT)))

        self.puck_markers = markers(frame_count)

        cv2.namedWindow(self.window_name, cv2.WINDOW_FREERATIO |
                        cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.window_name, self.on_click)
        cv2.createTrackbar(self.slider_name, self.window_name,
                           1, frame_count, self.on_trackbar)

    def release(self):
        cv2.destroyAllWindows()
        self.vs.release()
        self.puck_markers.release()

    def on_trackbar(self, pos, data=None):
        cv2.VideoCapture.set(self.vs, int(cv2.CAP_PROP_POS_FRAMES), pos)
        self.frame_index = pos
        self.process_frame()

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.puck_markers.pos_white[self.frame_index] = np.int0(
                np.asarray([x, y]))

        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self.puck_markers.pos_white[self.frame_index] = np.int0(
                    np.asarray([x, y]))
            else:
                self.puck_markers.pos_green[self.frame_index] = np.int0(
                    np.asarray([x, y]))

        print(x, y)
        self.puck_markers.draw(self.frame_index, self.frame_current)
        cv2.imshow(self.window_name, self.frame_current)

    def process_frame(self):
        _, self.frame_current = self.vs.read()
        cv2.setTrackbarPos(
            self.slider_name, self.window_name, self.frame_index)
        self.puck_markers.draw(self.frame_index, self.frame_current)
        cv2.imshow(self.window_name, self.frame_current)

    def increment_frame(self):
        self.frame_index += 1

    def decrement_frame(self):
        self.frame_index -= 1
        cv2.VideoCapture.set(self.vs, int(
            cv2.CAP_PROP_POS_FRAMES), self.frame_index)


def main():
    output = ui('out.mp4')

    while(True):
        output.process_frame()
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('['):  # left
            output.decrement_frame()
        else:
            output.increment_frame()

    output.release()


if __name__ == "__main__":
    main()
