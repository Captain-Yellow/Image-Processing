# This Python file uses the following encoding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

class clacHist2d:
    def __init__(self):
        print("calcHist2d")

    def calculateHistogram(self):
        img = cv2.imread(sys.argv[1])
        hisBins = [int(sys.argv[2]) if len(sys.argv[2]) > 0 and int(sys.argv[2]) > 0 else 256]
        fig = plt.figure()
#        histr = cv2.calcHist([img],[0],None,[256],[0,256])
        plt.hist(img.ravel(), bins=hisBins[0], range=[0, 256])
#        plt.hist(img.ravel(), bins=hisBins, range=[0, 256])

#        plt.plot(histr)
#        plt.show()

if __name__ == '__main__':
    p1 = clacHist2d()
    p1.calculateHistogram()
    plt.show()
