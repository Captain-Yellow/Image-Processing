# This Python file uses the following encoding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys


class CalcHist:
    def __init__(self):
        print("CalcHist")

    def calculateHistogram3d(self):
        img = cv2.imread(sys.argv[1])
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]
        # b,g,r = cv2.split(img);
        # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # r,g,b = cv2.split(hsv)
        fig = plt.figure()
        hisBins = [int(sys.argv[2]) if len(sys.argv[2]) > 0 and int(sys.argv[2]) > 0 else 256]
        ax = fig.add_subplot(111, projection='3d')
        # fig = plt.subplots(1)
        print(hisBins)
        for x, c, z in zip([r, g, b], ['r', 'g', 'b'], [30, 20, 10]):
            xs = np.arange(256)
            ys = cv2.calcHist([x], [0], None, hisBins, [0, 256])
            cs = [c] * len(xs)
            cs[0] = 'c'
            ax.bar(xs, ys.ravel(), zs=z, zdir='y', color=cs, alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # fig = plt.hist(img.ravel(), bins=hisBins, range=[0, 256])
        plt.show()




if __name__ == '__main__':
    p1 = CalcHist()
    p1.calculateHistogram3d()
