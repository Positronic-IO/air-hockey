""" Calculate the offset for yolo """
import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np
from redis import StrictRedis

redis = StrictRedis()

# Loggings
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Opencv valid photo extensions
EXTENSIONS = set([".bmp", ".jpeg", ".jpg", ".png", ".tiff"])


def load_img(path):
    """ Load imgage """
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (640, 480))
    return img


def find_table(img):
    """ Find the table and draw a bounding box """

    # isolate & enhance the HUE channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1:] = 0
    t, hue = cv2.threshold(hsv, 50, 255, cv2.THRESH_BINARY)
    hue = ~hue[:, :, 0]

    # denoise
    kernel = np.ones((5, 50))
    opened = cv2.morphologyEx(hue, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((10, 10)))

    # get the largest contiguous blob
    laplacian = cv2.Laplacian(closed, cv2.CV_8UC1)
    contours, hierarchy = cv2.findContours(
        laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)

    # square it off
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def calculate_offset(table, resolution):
    """ Calculate table offset """

    # Grab corners of table
    (tl, bl, br, tr) = table

    dimensions = {
        "l": int(max(tr[0], br[0]) - min(tl[0], bl[0])),
        "w": int(max(tl[1], tr[1]) - min(bl[1], br[1]))
    }

    # The goal is to have the top left corner of the table be (0,0).
    x_displacement = min(tl[0], bl[0])
    y_displacement = resolution[0] - min(tl[1], tr[1])

    # x, y offset
    offset = {"x": str(x_displacement), "y": str(y_displacement)}
    redis.set("table_offset", json.dumps(offset))
    redis.set("table_dimensions", json.dumps(dimensions))
    return offset, dimensions


if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(
        description='Find table and calculate pixel offset')
    parser.add_argument('--file', metavar='-f', type=str,
                        help='name of file of table')
    args = parser.parse_args()
    parsed_args = vars(args)

    # Parse image path
    img_path = parsed_args["file"]

    # Make sure we have a file
    if not img_path:
        logger.info("Please enter a valid file path to a image")
        sys.exit()

    # Make sure the file is a valid file type
    _, file_extension = os.path.splitext(img_path)

    if file_extension not in EXTENSIONS:
        logger.info(f"Extension {file_extension} is not supported.")
        sys.exit()

    # Load image
    img = load_img(img_path)

    # Create bounding box around table
    table = find_table(img)

    # Find dimensions of image, needed to calculate offset
    resolution = img.shape  # Note that these are in (y,x)

    # Calculate and publish offset to redis
    offset, dimensions = calculate_offset(table, resolution)
    logger.info(f"Table offset: {offset}")
    logger.info(f"Table dimensions: {dimensions}")
