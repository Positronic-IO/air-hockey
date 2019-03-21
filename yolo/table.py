import json
import logging
import math
import random
from ctypes import *

import cv2
import numpy as np
from skimage import draw, io
from imutils.video import WebcamVideoStream

import sys
import os
sys.path.append(os.getcwd())
from state_machine import AirHockeyTableState


state = AirHockeyTableState()

netMain = None
metaMain = None
altNames = None
thresh = 0.75
configPath = "./yolo/config/tiny.cfg"
weightPath = "./yolo/config/tiny.weights"
metaPath = "./yolo/config/tiny.data"

live = False
rotate = 0
black_background = False
full_screen = False
vs = None
predict_image = None
get_network_boxes = None
do_nms_sort = None
free_detections = None


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def init():
    global live, vs, predict_image, get_network_boxes, do_nms_sort, free_detections
    lib = CDLL("./yolo/darknet.so", RTLD_GLOBAL)
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [
        c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
    get_network_boxes.restype = POINTER(DETECTION)

    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p]
    make_network_boxes.restype = POINTER(DETECTION)

    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    network_predict = lib.network_predict
    network_predict.argtypes = [c_void_p, POINTER(c_float)]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    if netMain is None:
        netMain = load_net(configPath.encode("ascii"),
                           weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    if live:
        vs = WebcamVideoStream(src=0).start()
    else:
        vs = cv2.VideoCapture('yolo/data/data.mp4')


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, custom_image_bgr, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
#    custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)

    im, arr = array_to_image(custom_image)

    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                res.append(detection_to_puck(
                    (nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h))))

#    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res


def detection_to_puck(detection):
    label = detection[0]
    confidence = detection[1]
    bounds = detection[2]
    yExtent = int(bounds[3])
    xEntent = int(bounds[2])
    xCoord = int(bounds[0] - bounds[2]/2)
    yCoord = int(bounds[1] - bounds[3]/2)
    boundingBox = [
        [xCoord, yCoord],
        [xCoord, yCoord + yExtent],
        [xCoord + xEntent, yCoord + yExtent],
        [xCoord + xEntent, yCoord]
    ]
    return label, confidence, boundingBox


def get_frame():
    global live, vs, netMain, metaMain, thresh, rotate
    if live:
        frame = vs.read()
    else:
        _, frame = vs.read()

    if rotate != 0:
        rows, cols = frame.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotate, 1)
        frame = cv2.warpAffine(frame, M, (cols, rows))

    detections = detect(netMain, metaMain, frame, thresh)
    return detections, frame


def find_center_box(detections):
    """ Find center of bounding box (scaled)"""

    try:
        offset = json.loads(state.redis.get("table_offset"))
    except TypeError:
        offset = {"x": 0, "y": 0}

    # Run when there are detections
    try:
        data = detections[0]
        coords = data[2]
        center = {
            "x": int((coords[0][0] + coords[3][0]) / 2) - int(offset["x"]),
            "y": int((coords[0][1] + coords[1][1]) / 2) - int(offset["y"])
        }
    except IndexError:
        return None

    state.publish(name="puck", data=center)
    return center


def main():
    global black_background
    init()

    window_name = "science!"
    if full_screen:
        cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # crop_baseline = [120, 140, 80, 215]
    crop_baseline = [0, 0, 0, 0]
    crop = crop_baseline

    while(True):
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        detections, frame = get_frame()

        find_center_box(detections)
        # print(detections)
        # print(center)

        if frame is None:
            continue

        image = np.zeros_like(frame) if black_background else frame

        for (label, confidence, boundingBox) in detections:
            rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [
                                            x[0] for x in boundingBox], shape=image.shape)
            boxColor = (int(255 * (1 - (confidence ** 2))),
                        int(255 * (confidence ** 2)), 0)
            draw.set_color(image, (rr, cc), boxColor, alpha=0.8)

        image = image[crop[0]:image.shape[0] -
                      crop[1], crop[2]:image.shape[1] - crop[3]]
        image = cv2.resize(image, (1280, 800), interpolation=cv2.INTER_CUBIC)
        cv2.imshow(window_name, image)

    cv2.destroyAllWindows()
    if live:
        vs.stop()
    else:
        vs.release()

    return


if __name__ == "__main__":
    main()
