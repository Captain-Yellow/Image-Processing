#include "imageprocessing.h"

imageProcessing::imageProcessing() { }

/**##########################################
 *          Hough Transform
 * ##########################################
*/
std::pair<cv::Mat, cv::Mat> imageProcessing::cvHoughTransform(cv::Mat BWphoto) {

    // Declare the output variables
    cv::Mat dst, cdst, cdstP;

    // const char* default_file = "sudoku.png";
    // const char* filename = argc >=2 ? argv[1] : default_file;
    // Loads an image
    cv::Mat src = BWphoto;// imread( samples::findFile( filename ), IMREAD_GRAYSCALE );
    // Check if image is loaded fine
    if(src.empty()){
        printf("Error opening image\n");
        // printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        // return -1;
    }

    // Edge detection
    Canny(src, dst, 50, 200, 3);

    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, cv::COLOR_GRAY2BGR);
    cdstP = cdst.clone();

    // Standard Hough Line Transform
    std::vector<cv::Vec2f> lines; // will hold the results of the detection
    std::vector<int> threshold {180, 150, 100, 70, 50};
    std::vector<int> linThickness {1, 2, 3};
    HoughLines(dst, lines, 1, CV_PI/180, threshold[3], 0, 0 ); // runs the actual detection

    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( cdst, pt1, pt2, cv::Scalar(0,0,255), linThickness[1], cv::LINE_AA);
    }
    // Probabilistic Line Transform
    std::vector<cv::Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI/180, threshold[4], 50, 10 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        cv::Vec4i l = linesP[i];
        line( cdstP, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), linThickness[1], cv::LINE_AA);
    }
    // Show results
    // imshow("Source", src);
    // imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    // imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

    std::pair<cv::Mat, cv::Mat> houghTransform{cdst, cdstP};
    return houghTransform;
    //return cdst;
}

/**##########################################
 *    Generate Histogram gray and color
 * ##########################################
*/

cv::Mat imageProcessing::generateHistogram(cv::Mat orgImg, bool cvShow) {

    if (orgImg.channels() < 2) {
        cv::Mat image = orgImg;

        // allcoate memory for no of pixels for each intensity value
        int histogram[256];

        // initialize all intensity values to 0
        for(int i = 0; i < 255; i++) {
            histogram[i] = 0;
        }

        // calculate the no of pixels for each intensity values
        for(int y = 0; y < image.rows; y++)
            for(int x = 0; x < image.cols; x++)
                histogram[(int)image.at<uchar>(y,x)]++;

//        for(int i = 0; i < 256; i++)
//                qDebug() <<histogram[i]<<" ";

        // draw the histograms
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound((double) hist_w/256);

        cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(255, 255, 255));

        // find the maximum intensity element from histogram
        int max = histogram[0];
        for(int i = 1; i < 256; i++) {
            if(max < histogram[i]){
                max = histogram[i];
            }
        }

        // normalize the histogram between 0 and histImage.rows

        for(int i = 0; i < 255; i++) {
            histogram[i] = ((double)histogram[i]/max)*histImage.rows;
        }

        // draw the intensity line for histogram
        for(int i = 0; i < 255; i++) {
            line(histImage, cv::Point(bin_w*(i), hist_h),
                                  cv::Point(bin_w*(i), hist_h - histogram[i]),
                 cv::Scalar(0,0,0), 1, 8, 0);
        }

        if(cvShow == 1) {
            cv::namedWindow("histogram", cv::WINDOW_AUTOSIZE);
            imshow("histogram", histImage);
            cv::waitKey(0);
        }



        //claculating image histogram
        int narrays = 1;
        int channels[] = { 0 };
        cv::Mat grayHistValue;
        int dims = 1;
        int histSize[] = { 256 };
        float hranges[] = { 0.0, 255.0 };
        const float *ranges[] = { hranges };
        calcHist(&orgImg, narrays, channels, cv::Mat(), grayHistValue, dims, histSize, ranges);
        float sum = 0;
        for (int i = 0; i < 256; i++) {
            // qDebug() << "value " << i << " = " << grayHistValue.at<float>(i)<< "\n";
            sum = sum + grayHistValue.at<float>(i);
        }
        // return grayHistValue;
        double maxVal = 0;
        double minVal = 0;
        minMaxLoc(grayHistValue, &minVal, &maxVal, 0, 0);
        int hpt = static_cast<int>(0.9*histSize[0]);
        cv::Mat histImg(histSize[0], histSize[0], CV_8UC1, cv::Scalar(255));
        for (int h = 0; h < histSize[0]; h++) {
            float binVal = (grayHistValue.at<float>(h)*hpt)/maxVal;
            line(histImg, cv::Point(h, histSize[0]), cv::Point(h, histSize[0] - binVal), cv::Scalar::all(0));
        }
        if(cvShow == 1) {
            cv::namedWindow("histogram", cv::WINDOW_AUTOSIZE);
            imshow("histogram", histImg);
            cv::waitKey(0);
        }



        return histImage;
    }
    else if (orgImg.channels() == 3) {
//        //claculating image histogram
//        int narrays = 1;
//        int chanels[] = { 0,1,2 };
//        cv::Mat colorHistValue;
//        int dims = 3;
//        int histSize[] = { 10,10,10 };
//        float hranges[] = { 0, 255 };
//        const float* ranges[] = { hranges, hranges, hranges };

//        calcHist(&img, narrays, chanels, cv::Mat(), colorHistValue, dims, histSize, ranges);

//        for (int i = 0; i < 10; i++)
//            for (int j = 0; j < 10; j++)
//                for (int k = 0; k< 10; k++)
//                    qDebug() << "value " << i <<"," <<j <<"," <<k << "=" << colorHistValue.at<float>(i,j,k) << "\n";

//        return colorHistValue;
//        cv::waitKey(0);



        std::vector<cv::Mat> bgr_planes;
          split( orgImg, bgr_planes );

          int histSize = 256;

          float range[] = { 0, 256 } ;
          const float* histRange = { range };

          bool uniform = true; bool accumulate = false;

          cv::Mat b_hist, g_hist, r_hist;

          calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
          calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
          calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

          // Draw the histograms for B, G and R
          int hist_w = 512; int hist_h = 400;
          int bin_w = cvRound( (double) hist_w/histSize );

          cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 255,255,255) );

          normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
          normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
          normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

          for( int i = 1; i < histSize; i++ ) {
              line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                               cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                               cv::Scalar( 255, 0, 0), 2, 8, 0  );
              line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                               cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                               cv::Scalar( 0, 255, 0), 2, 8, 0  );
              line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                               cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                               cv::Scalar( 0, 0, 255), 2, 8, 0  );
          }
          if(cvShow == 1) {
              namedWindow("calcHist Demo", cv::WINDOW_AUTOSIZE );
              imshow("calcHist Demo", histImage );
              cv::waitKey(0);
          }
          return histImage;
    }
}

cv::Mat imageProcessing::HistogramEqualizer(cv::Mat src) {
    if (src.channels() < 2) {
        cv::equalizeHist(src, dst);
        return dst;
    }
    else if (src.channels() == 3) {
        /**
         * OpenCV loads color images in BGR color space. With this color space, it is not possible to equalize
         * the histogram without affecting to the color information because all 3 channels contain
         * color information. Therefore you have to convert the BGR image to a color space like YCrCb.
         * In YCrCb color space, the Y channel of the image only contains intensity information where as Cr and Cb channels
         * contain all the color information of the image. Therefore only the Y channel should be processed to
         * get a histogram equalized image without changing any color information. After the processing,
         * the YCrCb image should be converted back to the BGR color space before calling imshow() function.
         */
        //Convert the image from BGR to YCrCb color space
        cvtColor(src, dst, cv::COLOR_BGR2YCrCb);

        //Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
        std::vector<cv::Mat> vec_channels;
        split(dst, vec_channels);

        //Equalize the histogram of only the Y channel
        equalizeHist(vec_channels[0], vec_channels[0]);

        //Merge 3 channels in the vector to form the color image in YCrCB color space.
        merge(vec_channels, dst);

        //Convert the histogram equalized image from YCrCb to BGR color space again
        cvtColor(dst, dst, cv::COLOR_YCrCb2BGR);

        return dst;
    }
}

/**##########################################
 *                Noise
 * ##########################################
*/
cv::Mat imageProcessing::noise_saltAndPepper(cv::Mat srcImg, int num) {
//    cv::Mat saltAndPepperImage;
    int i, j, x, y;
    for (int k=0; k<num; k++) {
        // rand() is the MFC random number generator
        i= rand()%srcImg.cols;
        j= rand()%srcImg.rows;
        x= rand()%srcImg.cols;
        y= rand()%srcImg.rows;
        //qDebug() << i<<j<<x<<y;
        if (srcImg.channels() == 1) { // gray-level image
            srcImg.at<uchar>(j,i)= 255;
            srcImg.at<uchar>(y,x)= 0;
        } else if (srcImg.channels() == 3) { // color image
            srcImg.at<cv::Vec3b>(j,i)[0]= 255;
            srcImg.at<cv::Vec3b>(j,i)[1]= 255;
            srcImg.at<cv::Vec3b>(j,i)[2]= 255;
            srcImg.at<cv::Vec3b>(y,x)[0]= 0;
            srcImg.at<cv::Vec3b>(y,x)[1]= 0;
            srcImg.at<cv::Vec3b>(y,x)[2]= 0;
        }
    }
    return srcImg;
}

cv::Mat imageProcessing::noise_gaussian(cv::Mat srcImg, float mean, float sigma) {
//    cv::Mat gaussian_noise = srcImg.clone();
//    randn(gaussian_noise, 128, 30);
    cv::Mat noise(srcImg.size(),srcImg.type());
    //float m = (10, 12, 34);
    //float sigma = (1, 5, 50);
    cv::randn(noise, mean, sigma); //mean and variance
    srcImg += noise;

    return srcImg;
}

cv::Mat imageProcessing::noise_impulse(cv::Mat srcImg, int num) {
    int i, j;
    for (int k=0; k<num; k++) {
        // rand() is the MFC random number generator
        i= rand()%srcImg.cols;
        j= rand()%srcImg.rows;
        //qDebug() << i<<j<<x<<y;
        if (srcImg.channels() == 1) { // gray-level image
            srcImg.at<uchar>(j,i)= rand()%255;
        } else if (srcImg.channels() == 3) { // color image
            srcImg.at<cv::Vec3b>(j,i)[0]= rand()%255;
            srcImg.at<cv::Vec3b>(j,i)[1]= rand()%255;
            srcImg.at<cv::Vec3b>(j,i)[2]= rand()%255;
        }
    }
    return srcImg;
}

/**##########################################
 *            Convolution
 * ##########################################
*/
cv::Mat imageProcessing::convolution2d(cv::Mat src, cv::Mat mask) {
    cv::Mat result;
    //apply convolution filter. The third parameter (depth) is set to -1, which means the bit-depth of the output image
    //is the same as the input image. So if the input image is of type CV_8UC3, the output image will also be of the same
    //type. Later we will see certain kinds of kernels where CV_32F or CV_64F should be used.
    cv::filter2D(src, result, -1, mask, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    return result;
}

cv::Mat imageProcessing::sepConvolution2d(cv::Mat src, cv::Mat Xmask, cv::Mat Ymask) {
    cv::Mat result;
    cv::sepFilter2D(src, result, src.depth(), Xmask, Ymask, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    return result;
}

/**##########################################
 *               Filtering
 * ##########################################
*/

cv::Mat imageProcessing::gaussianFilter(cv::Mat srcImg, int kernelSize) {
    cv::GaussianBlur( srcImg, dst, cv::Size( kernelSize, kernelSize ), 0, 0 );
    return dst;
}

cv::Mat imageProcessing::medianFilter(cv::Mat srcImg, int kernelSize) {
    cv::medianBlur ( srcImg, dst, kernelSize );
    return dst;
}

cv::Mat imageProcessing::NormalizedBlockFilter(cv::Mat src, int kernelSize) {
    cv::blur( src, dst, cv::Size( kernelSize, kernelSize ), cv::Point(-1,-1) );
    return dst;
}

cv::Mat imageProcessing::BilateralFilter(cv::Mat src, int kernelSize) {
    cv::bilateralFilter ( src, dst, kernelSize, kernelSize*2, kernelSize/2 );
    return dst;
}

/**##########################################
 *             segmentation
 * ##########################################
*/

cv::Mat imageProcessing::threshold_Segmentaion(cv::Mat src) {
    cv::threshold(src, dst, 0, 255.0, cv::THRESH_BINARY+cv::THRESH_OTSU);
//    cv::inRange(src, cv::Scalar(0, 125, 0), cv::Scalar(255, 200, 255), dst);

    return dst;
}

cv::Mat imageProcessing::inrange_Segmentaion(cv::Mat src, int RL, int GL, int BL, int RU, int GU, int BU) {
    auto lower_color_bounds = cv::Scalar(BL, GL, RL);  // cv::Scalar(20, 0, 0)
    auto upper_color_bounds = cv::Scalar(BU, GU, RU);  // cv::Scalar(100,255,100)

    cv::Mat src1 = src.clone();
    cv::Mat gray;
    cv::cvtColor(src1, gray, cv::COLOR_BGR2GRAY, 0);
    cv::Mat mask;
    inRange(src1, lower_color_bounds, upper_color_bounds, mask);
    cv::Mat mask_rgb;
    cvtColor(mask, mask_rgb, cv::COLOR_GRAY2BGR, 0);
    src1 = src1 & mask_rgb;

    return src1;
}

cv::Mat imageProcessing::kmean_Segmentaion(cv::Mat source, int k, int d) {
    if (d == 3) {
        const int MAX_ITERATIONS = 2;
        const unsigned int singleLineSize = source.rows * source.cols;
        cv::Mat data = source.reshape(1, singleLineSize);
        data.convertTo(data, CV_32F);
        std::vector<int> labels;
        cv::Mat1f colors;
        cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.), MAX_ITERATIONS, cv::KMEANS_PP_CENTERS, colors);
        for (unsigned int i = 0 ; i < singleLineSize ; i++ ){
                    data.at<float>(i, 0) = colors(labels[i], 0);
                    data.at<float>(i, 1) = colors(labels[i], 1);
                    data.at<float>(i, 2) = colors(labels[i], 2);
        }
        cv::Mat outputImage = data.reshape(3, source.rows);
        outputImage.convertTo(outputImage, CV_8U);
        return outputImage;
    }
    else if (d == 1) {
        const unsigned int singleLineSize = source.rows * source.cols;
        cv::Mat data = source.reshape(1, singleLineSize);
        data.convertTo(data, CV_32F);
        std::vector<int> labels;
        cv::Mat1f colors;
        cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.), 2, cv::KMEANS_PP_CENTERS, colors);
        for (unsigned int i = 0; i < singleLineSize; i++) {
            data.at<float>(i) = colors(labels[i]);
        }
        cv::Mat outputImage = data.reshape(1, source.rows);
        outputImage.convertTo(outputImage, CV_8U);
        return outputImage;
    }
}

cv::Mat imageProcessing::labeling_Morphology(cv::Mat src) {
    cv::Mat stats, centroids, labelImage;
    int nLabels = connectedComponentsWithStats(src, labelImage, stats, centroids, 8, CV_32S);
    cv::Mat mask(labelImage.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat surfSup=stats.col(4)>2000;

    for (int i = 1; i < nLabels; i++) {
        if (surfSup.at<uchar>(i, 0)) {
            mask = mask | (labelImage==i);
        }
    }
    cv::Mat r(src.size(), CV_8UC1, cv::Scalar(0));
    src.copyTo(r,mask);

    return r;
}
/**##########################################
        Morphological Transformations
 * ##########################################
*/

cv::Mat imageProcessing::Erosion_Morphology(cv::Mat src, cv::Mat kernel) {
    cv::erode(src, dst, kernel, cv::Point(-1,-1), 1);
    return dst;
}

cv::Mat imageProcessing::dilation_Morphology(cv::Mat src, cv::Mat kernel) {
    cv::dilate(src, dst, kernel, cv::Point(-1,-1), 1);
    return dst;
}

cv::Mat imageProcessing::opening_Morphology(cv::Mat src, cv::Mat kernel) {
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), 1);
    return dst;
}

cv::Mat imageProcessing::closing_Morphology(cv::Mat src, cv::Mat kernel) {
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 1);
    return dst;
}

/**##########################################
 *          Transform and filter
 * ##########################################
*/

// lowPass Filter
void imageProcessing::lowpassFilter(const cv::Mat &dft_Filter, int distance) {
    cv::Mat tmp = cv::Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

    cv::Point centre = cv::Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
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

    cv::Mat toMerge[] = {tmp, tmp};
    merge(toMerge, 2, dft_Filter);
}

cv::Mat imageProcessing::computeDFT(cv::Mat image) {
    cv::Mat padded;                            //expand input image to optimal size
    int m = cv::getOptimalDFTSize( image.rows );
    int n = cv::getOptimalDFTSize( image.cols ); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex;
    merge(planes, 2, complex);         // Add to the expanded another plane with zeros
    dft(complex, complex, cv::DFT_COMPLEX_OUTPUT);  // fourier transform
    return complex;
}

void imageProcessing::fftShift(cv::Mat magI) {
    // crop if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    int cx = magI.cols/2;
    int cy = magI.rows/2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

/**##########################################
 *              Edge Detector
 * ##########################################
*/

cv::Mat imageProcessing::Sobel_EdgeDetector(cv::Mat src1, QString axis) {
    cv::Mat image = src1.clone(),src, src_gray;
    cv::Mat grad;
    int ddepth = CV_16S;
    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur(image, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    // Convert the image to grayscale
    cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, ddepth, 1, 0, 5, 1, 0, cv::BORDER_DEFAULT);
    Sobel(src_gray, grad_y, ddepth, 0, 1, 5, 1, 0, cv::BORDER_DEFAULT);
    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    return grad;
}

cv::Mat imageProcessing::Canny_EdgeDetector(cv::Mat src) {
    cv::Mat img = src.clone();

    // Convert to graycsale
    cv::Mat img_gray;
    if(img.channels() > 2) {
        cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    }
    else
        img_gray = img;
    // Blur the image for better edge detection
    cv::Mat img_blur;
    GaussianBlur(img_gray, img_blur, cv::Size(3,3), 0);

    // Canny edge detection
    cv::Mat edges;
    Canny(img_blur, edges, 100, 200, 3, false);
    return edges;
}

/**##########################################
 *                encoding
 * ##########################################
*/

void imageProcessing::HuffmanCoding() {

}
