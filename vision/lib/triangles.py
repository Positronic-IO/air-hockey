import numpy as np

_windowSize = 50
_all = {
    "top": {
        "x": np.array([]),
        "y": np.array([])
    },
    "left": {
        "x": np.array([]),
        "y": np.array([])
    },
    "right": {
        "x": np.array([]),
        "y": np.array([])
    }
}

def append(triangle):

    top = triangle[0]
    if triangle[1][0][1] < top[0][1]:
        top = triangle[1]
    if triangle[2][0][1] < top[0][1]:
        top = triangle[2]
    
    left = triangle[0]
    if triangle[1][0][0] < left[0][0]:
        left = triangle[0]
    if triangle[2][0][0] < left[0][0]:
        left = triangle[2]

    right = triangle[0]
    if triangle[1][0][0] > right[0][0]:
        right = triangle[1]
    if triangle[2][0][0] > right[0][0]:
        right = triangle[2]

    _all["top"]["x"] = np.append(_all["top"]["x"], top[0][0])
    _all["top"]["y"] = np.append(_all["top"]["y"], top[0][1])
    _all["left"]["x"] = np.append(_all["left"]["x"], left[0][0])
    _all["left"]["y"] = np.append(_all["left"]["y"], left[0][1])
    _all["right"]["x"] = np.append(_all["right"]["x"], right[0][0])
    _all["right"]["y"] = np.append(_all["right"]["y"], right[0][1])

    _all["top"]["x"] = _all["top"]["x"][-_windowSize:]
    _all["top"]["y"] = _all["top"]["y"][-_windowSize:]
    _all["left"]["x"] = _all["left"]["x"][-_windowSize:]
    _all["left"]["y"] = _all["left"]["y"][-_windowSize:]
    _all["right"]["x"] = _all["right"]["x"][-_windowSize:]
    _all["right"]["y"] = _all["right"]["y"][-_windowSize:]


def average():
    average = np.array([
        [np.average(_all["top"]["x"]), np.average(_all["top"]["y"])],
        [np.average(_all["left"]["x"]), np.average(_all["left"]["y"])],
        [np.average(_all["right"]["x"]), np.average(_all["right"]["y"])]
    ], dtype=np.int32)

    return average    