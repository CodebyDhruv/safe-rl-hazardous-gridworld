import numpy as np
from minigrid.core.world_object import WorldObj

class Hazard(WorldObj):
    def __init__(self):
        super().__init__("floor", "blue")

    def can_overlap(self):
        return True

    def render(self, img):
        img[:] = (255, 165,0)  
        h, w, _ = img.shape
        thickness = 2
        for i in range(h):
            img[i, i] = (0, 0, 0)
            img[i, w - i - 1] = (0, 0, 0)
            
            for t in range(1, thickness):
                if i + t < h:
                    img[i + t, i] = (0, 0, 0)
                    img[i + t, w - i - 1] = (0, 0, 0)
