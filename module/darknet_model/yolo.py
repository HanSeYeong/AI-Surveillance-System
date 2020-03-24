from ctypes import Structure, c_float, c_int, POINTER, c_void_p, pointer, RTLD_GLOBAL, c_char_p, windll
import ctypes
import math
import random
import os
import cv2
import numpy as np
import sys


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


hasGPU = True
windll.kernel32.SetDllDirectoryW(None)
lib = ctypes.CDLL(os.getcwd() + "/module/darknet_model/model_data/yolo_cpp_dll.dll", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = \
    [c_void_p, c_int, c_int, c_float, c_float, POINTER(
        c_int), c_int, POINTER(c_int), c_int]
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

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

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


def array_to_image(arr):
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


def convertBack_int(x, y, w, h):
    x_ = int(round(x - (w / 2)))
    w_ = int(round(w))
    y_ = int(round(y - (h / 2)))
    h_ = int(round(h))
    return x_, y_, w_, h_


def convertBack(x, y, w, h):
    x_ = round(x - (w / 2), 3)
    w_ = round(w, 3)
    y_ = round(y - (h / 2), 3)
    h_ = round(h, 3)
    return x_, y_, w_, h_


def calcDistance(v1, v2):
    dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


def destroyLib():
    del lib


def detect(net, meta, altNames, image, ratio_w, ratio_h, thresh=.5, hier_thresh=.5, nms=.45):
    int_boxes = []
    names = []
    accuracies = []
    im, arr = array_to_image(image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h,
                             thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    do_nms_sort(dets, num, meta.classes, nms)
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0.5:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]

                if nameTag in ['person']:
                    x, y, w, h = convertBack_int(b.x * ratio_w, b.y * ratio_h, b.w * ratio_w, b.h * ratio_h)
                    int_boxes.append([x, y, w, h])
                    names.append(nameTag)
                    accuracies.append(dets[j].prob[i])

    free_detections(dets, num)

    return int_boxes, names, accuracies


netMain = None
metaMain = None
altNames = None


class YOLO:
    def __init__(self):
        global metaMain, netMain, altNames, lib
        configPath = "module/darknet_model/model_data/model/car.cfg"
        weightPath = "module/darknet_model/model_data/model/car_608.weights"
        metaPath = "module/darknet_model/model_data/model/car.data"

        self.netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
        self.metaMain = load_meta(metaPath.encode("ascii"))
        if altNames is None:
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
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        # self.Width = input_size[0]
        # self.Height = input_size[1]
        self.Width = 416
        self.Height = 416
        self.ratio_w = 1920 / self.Width
        self.ratio_h = 1080 / self.Height

    def detect_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.Width, self.Height), interpolation=cv2.INTER_LINEAR)
        return detect(self.netMain, self.metaMain, self.altNames, image,
                      self.ratio_w, self.ratio_h, thresh=0.25)

    def destroyDLL(self):
        sys._clear_type_cache()