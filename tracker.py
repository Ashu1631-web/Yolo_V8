import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self):
        self.nextObjectID = 0
        self.objects = {}

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.nextObjectID += 1

    def update(self, rects):
        if len(rects) == 0:
            return self.objects

        inputCentroids = []

        for (x1, y1, x2, y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids.append((cX, cY))

        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)
        else:
            self.objects = dict(enumerate(inputCentroids))

        return self.objects
