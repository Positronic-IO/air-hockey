import socket
import time
import numpy as np
class Robot:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.ipaddress = "192.168.4.1"
        self.port = 2222

        self.mode = "mm2"

        self.current_pos_x_low = chr(9)
        self.current_pos_x_high = chr(1)

        self.current_pos_y_low = chr(0)
        self.current_pos_y_high = chr(0)

        self.speed_low = chr(255)
        self.speed_high = chr(200)

        self.accel_low = chr(255)
        self.accel_high = chr(200)

        self.target_x_low = chr(0)
        self.target_x_high = chr(0)

        self.target_y_low = chr(0)
        self.target_y_high = chr(0)

    def goto(self, x, y, wait=False):

        self.target_x_low  = chr( x & 0xff)
        self.target_x_high = chr((x & 0xff00) >> 8)
        self.target_y_low  = chr( y & 0xff)
        self.target_y_high = chr((y & 0xff00) >> 8)
        payload = self.mode + self.target_x_high + self.target_x_low + self.target_y_high + self.target_y_low + self.speed_high + self.speed_low + self.accel_high + self.accel_low + self.current_pos_x_high + self.current_pos_x_low + self.current_pos_y_high + self.current_pos_y_low
        self.sock.sendto(payload.encode(), (self.ipaddress, self.port))
        if wait:
            old_x = ord(self.current_pos_x_high) * 255 + ord(self.current_pos_x_low)
            old_y = ord(self.current_pos_y_high) * 255 + ord(self.current_pos_y_low)
            wait_time = (abs(old_x - x) + abs(old_y - y)) * .0017 + .01 + 1
            time.sleep(wait_time)
        self.current_pos_x_high = self.target_x_high
        self.current_pos_x_low  = self.target_x_low
        self.current_pos_y_high = self.target_y_high
        self.current_pos_y_low  = self.target_y_low


    def setup_shot(self, puck):
        self.goto(puck[0], 50, True)
        self.goto(puck[0], 300, True)
        self.goto(265, 0, True)

    def setup_angled_shot(self, puck):
        target = (265, 700)
        vector = np.array((target[0] - puck[0], target[1] - puck[1]))
        unit_vector = vector / np.linalg.norm(vector)
        starting_point = (int(unit_vector[0] * -50) + puck[0], int(unit_vector[1] * -50) + puck[1])
        print(starting_point)
        self.goto(starting_point[0], starting_point[1], True)
        end_point = (int(unit_vector[0] * 150) + puck[0], int(unit_vector[1] * 150) + puck[1])
        end_point = (int((end_point[1] - 350) * unit_vector[1] / unit_vector[0] + end_point[0] ), 350)
        print(end_point)
        self.goto(end_point[0], end_point[1], True)
        self.goto(265, 0, True)