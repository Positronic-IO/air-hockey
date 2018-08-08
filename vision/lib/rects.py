import numpy as np

_windowSize = 50
_all = {
    "top": {
        "left": {
            "x": np.array([]),
            "y": np.array([])
        },
        "right": {
            "x": np.array([]),
            "y": np.array([])
        }
    },
    "bottom": {
        "left": {
            "x": np.array([]),
            "y": np.array([])
        },
        "right": {
            "x": np.array([]),
            "y": np.array([])
        }
    }
}

def append(rect):
    _all["top"]["left"]["x"] = np.append(_all["top"]["left"]["x"], rect[0][0])
    _all["top"]["left"]["y"] = np.append(_all["top"]["left"]["y"], rect[0][1])
    _all["bottom"]["left"]["x"] = np.append(_all["bottom"]["left"]["x"], rect[1][0])
    _all["bottom"]["left"]["y"] = np.append(_all["bottom"]["left"]["y"], rect[1][1])
    _all["bottom"]["right"]["x"] = np.append(_all["bottom"]["right"]["x"], rect[2][0])
    _all["bottom"]["right"]["y"] = np.append(_all["bottom"]["right"]["y"], rect[2][1])
    _all["top"]["right"]["x"] = np.append(_all["top"]["right"]["x"], rect[3][0])
    _all["top"]["right"]["y"] = np.append(_all["top"]["right"]["y"], rect[3][1])

    _all["top"]["left"]["x"] = _all["top"]["left"]["x"][-_windowSize:]
    _all["top"]["left"]["y"] = _all["top"]["left"]["y"][-_windowSize:]
    _all["bottom"]["left"]["x"] = _all["bottom"]["left"]["x"][-_windowSize:]
    _all["bottom"]["left"]["y"] = _all["bottom"]["left"]["y"][-_windowSize:]
    _all["bottom"]["right"]["x"] = _all["bottom"]["right"]["x"][-_windowSize:]
    _all["bottom"]["right"]["y"] = _all["bottom"]["right"]["y"][-_windowSize:]
    _all["top"]["right"]["x"] = _all["top"]["right"]["x"][-_windowSize:]
    _all["top"]["right"]["y"] = _all["top"]["right"]["y"][-_windowSize:]


def average():
    average = np.array([
        [np.average(_all["top"]["left"]["x"]), np.average(_all["top"]["left"]["y"])],
        [np.average(_all["bottom"]["left"]["x"]), np.average(_all["bottom"]["left"]["y"])],
        [np.average(_all["bottom"]["right"]["x"]), np.average(_all["bottom"]["right"]["y"])],
        [np.average(_all["top"]["right"]["x"]), np.average(_all["top"]["right"]["y"])]
    ], dtype=np.int32)

    return average    