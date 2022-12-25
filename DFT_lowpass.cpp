#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <iostream>

using namespace std;
using namespace cv;

// low Pass Filter
cv::Mat computeDFT(Mat image);
void fftShift(Mat magI);
void lowpassFilter(const cv::Mat &dft_Filter, int distance);


int main(int argc, char *argv[]){

    int radius=35;
    cv::Mat img, complexImg, filter, filterOutput, imgOutput, planes[2];

    img = imread("/home/muhammad/MyCodes/OpenCV/Images/Img/aibo_2005.jpg", 0);
    if(img.empty())
    {
        return -1;
    }

    complexImg = computeDFT(img);
    filter = complexImg.clone();

    lowpassFilter(filter, radius); // create an ideal low pass filter

    fftShift(complexImg); // rearrage quadrants
    mulSpectrums(complexImg, filter, complexImg, 0); // multiply 2 spectrums
    fftShift(complexImg); // rearrage quadrants

    // compute inverse
    idft(complexImg, complexImg);

    split(complexImg, planes);
    normalize(planes[0], imgOutput, 0, 1, CV_MINMAX);

    split(filter, planes);
    normalize(planes[1], filterOutput, 0, 1, CV_MINMAX);

    imshow("Input image", img);
    waitKey(0);
    imshow("Filter", filterOutput);
    waitKey(0);
    imshow("Low pass filter", imgOutput);
    waitKey(0);
    destroyAllWindows();

    return 0;

}

// create an ideal low pass filter
void lowpassFilter(const cv::Mat &dft_Filter, int distance)
{
    Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

    Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
    double radius;

    for(int i = 0; i < dft_Filter.rows; i++)
    {
        for(int j = 0; j < dft_Filter.cols; j++)
        {
            radius = (double) sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
            if(radius>distance){
                tmp.at<float>(i,j) = (float)0;
            }else{
                tmp.at<float>(i,j) = (float)1;
            }

        }
    }

    Mat toMerge[] = {tmp, tmp};
    merge(toMerge, 2, dft_Filter);
}


// Compute the Discrete fourier transform
cv::Mat computeDFT(Mat image) {
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols ); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complex;
    merge(planes, 2, complex);         // Add to the expanded another plane with zeros
    dft(complex, complex, DFT_COMPLEX_OUTPUT);  // fourier transform
    return complex;
}

void fftShift(Mat magI) {

    // crop if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
