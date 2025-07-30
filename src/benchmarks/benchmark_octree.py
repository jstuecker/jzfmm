import fmdj
import time
import sys

class Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, open("logs/octree.log", "a+"))

print("Starting Tests!")