import numpy as np
from timeit import default_timer as timer

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