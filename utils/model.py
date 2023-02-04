import onnxruntime as ort
import yaml
import numpy as np
# import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
from collections import namedtuple, OrderedDict


class yolov5:
    def __init__(self, modelpath, classpath, confThreshold=0.25, nmsThreshold=0.45):
        with open(classpath, errors='ignore', encoding='UTF-8') as f:
            info = yaml.safe_load(f)
        self.classes = info['names']
        self.num_classes = len(self.classes)
        # self.inpHeight, self.inpWidth = 640, 640
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(modelpath, providers=['CUDAExecutionProvider'])
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

    def forward(self, imgsrc):
        preds = self.net.run(None, {self.net.get_inputs()[0].name: imgsrc})

        return preds

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        im = np.float32(np.zeros(imgsz))  # input image
        self.forward(im)  # warmup