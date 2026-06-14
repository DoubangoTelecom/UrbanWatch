
import numpy as np, pathlib, os, ctypes
from sys import platform

class Rgb2LXY(object):
    
    def __init__(self):
        # load
        libpath = os.path.join(pathlib.Path(__file__).resolve().parent, '..', 'rgb2lxy.dll' if platform.startswith('win') else 'librgb2lxy.so')
        self.rgb2lxyLib = ctypes.cdll.LoadLibrary(libpath)
        self.rgb2lxyLib.process.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double]
        self.rgb2lxyLib.process.restype = ctypes.c_int
        # init
        self.rgb2lxyLib.init()

    def process(self, image_rgb : np.array):
        lxy = image_rgb.tobytes()
        height, width = image_rgb.shape[:2]
        self.rgb2lxyLib.process(lxy, width, height, width, 0.0)
        lxy = np.frombuffer(lxy, dtype=np.uint8)
        return lxy.reshape((height, width, 3))