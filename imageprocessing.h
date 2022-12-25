#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

//#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv2/core/cvstd.hpp>
#include <iostream>
#include <QDebug>

//using namespace std;
//using namespace cv;

class imageProcessing {

    private:
        cv::Mat dst;
        int MAX_KERNEL_LENGTH = 31;

    public:
        imageProcessing();
        std::pair<cv::Mat, cv::Mat> cvHoughTransform(cv::Mat);
        cv::Mat generateHistogram(cv::Mat, bool cvShow = 0);
        cv::Mat noise_saltAndPepper(cv::Mat, int num);
        cv::Mat noise_gaussian(cv::Mat srcImg, float mean, float sigma);
        cv::Mat noise_impulse(cv::Mat srcImg, int num);
        cv::Mat convolution2d(cv::Mat, cv::Mat);
        cv::Mat sepConvolution2d(cv::Mat, cv::Mat, cv::Mat);
        cv::Mat gaussianFilter(cv::Mat, int kernelSize = 31);
        cv::Mat medianFilter(cv::Mat, int kernelSize = 31);
        cv::Mat NormalizedBlockFilter(cv::Mat, int kernelSize = 31);
        cv::Mat BilateralFilter (cv::Mat, int kernelSize = 31);
        cv::Mat HistogramEqualizer(cv::Mat);
        cv::Mat threshold_Segmentaion(cv::Mat);
        cv::Mat inrange_Segmentaion(cv::Mat, int RL, int GL, int BL, int RU, int GU, int BU);
        cv::Mat kmean_Segmentaion(cv::Mat, int, int);
        cv::Mat Erosion_Morphology(cv::Mat, cv::Mat);
        cv::Mat dilation_Morphology(cv::Mat, cv::Mat);
        cv::Mat opening_Morphology(cv::Mat, cv::Mat);
        cv::Mat closing_Morphology(cv::Mat, cv::Mat);
        cv::Mat labeling_Morphology(cv::Mat);
        cv::Mat Sobel_EdgeDetector(cv::Mat, QString);
        cv::Mat Canny_EdgeDetector(cv::Mat);

        void lowpassFilter(const cv::Mat&, int);
        cv::Mat computeDFT(cv::Mat);
        void fftShift(cv::Mat);

        void HuffmanCoding();


};

#endif // IMAGEPROCESSING_H
