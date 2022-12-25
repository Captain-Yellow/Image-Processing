#ifndef MAINWINDOW_H
#define MAINWINDOW_H


#include <QDebug>
#include <QScreen>
//#include <QFile>
#include <QFileDialog>
#include <QApplication>
#include <QMainWindow>
#include <QMessageBox>
#include <QRandomGenerator>
#include <QProcess>
#include <QStatusBar>
#include <QRegularExpression>
//#include <QtGlobal>

//#include <qt5/QtCharts/qchartglobal.h>
//#include <qt5/QtCharts/qbarset.h>
//#include <qt5/QtCharts/qchartview.h>
//#include <qt5/QtCharts/qbarseries.h>
//QT_CHARTS_USE_NAMESPACE

#include "imageprocessing.h"


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

    public:
        MainWindow(QWidget *parent = nullptr);
        ~MainWindow();
        static bool saveFlage;
        static QString colorImgType;
        bool pathCheck(void);
        void loadImageFromPath(QString);
        void loadHistogram(cv::Mat);
        void imgTypeCheck(int*, cv::Mat);
        void histogramPresenter();
        void photoPresenter(cv::Mat , QString currentType);
        cv::Mat changeLayer(cv::Mat, QString currentType = colorImgType);
        void ShowMessage(QString, QString);
        cv::Mat get_StructureElement_2d_odd();
        void colorChange(cv::Mat&, QString, QString);
        // cv::Mat loadFromQrc(QString qrc, int flag);

    private slots:
        void on_BT_Gray_clicked();
        void on_BT_Color_clicked();
        void on_BT_OpenPath_clicked();
        void on_BT_Save_clicked();
        void on_BT_Binary_clicked();
        void on_BT_houghTransform_clicked();
        void on_BT_Orginal_clicked();
        void on_BT_MyHoughTransform_clicked();
        void on_BT_probabilisticHoughTransform_clicked();
        void on_BT_GenenrateHistogram_clicked();
        void on_BT_calcHist3d_clicked();
        void on_BT_AddNoise_clicked();
        void on_BT_convolve2d_clicked();
        void on_BT_Filter_clicked();
        void on_BT_equalizeHistogram_clicked();
        void on_BT_changeColorChanels_clicked();
        void on_BT_CVHistogram_clicked();
        void on_BT_huffmanCoding_clicked();
        void on_BT_ThresholdingSegmentation_clicked();
        void on_BT_1DConvolve_clicked();
        void on_BT_ClusteringSegmentation_clicked();
        void on_BT_ThresholdingSegmentation_2_clicked();
        void on_BT_ThrSeg_orgAndT1_clicked();
        void on_BT_ThrSeg_orgAndT2_clicked();
        void on_BT_ThrSeg_T1AndT2_clicked();
        void on_BT_erosion_Morphology_clicked();
        void on_BT_dilation_Morphology_clicked();
        void on_BT_opening_Morphology_clicked();
        void on_BT_closing_Morphology_clicked();
        void on_BT_labeling_morphology_clicked();
        void on_BT_DFTtransform_clicked();
        void on_BT_SobelEdgeDetector_clicked();
        void on_BT_CannyEdgeDetector_clicked();
        void on_BT_PerwittEdgeDetector_clicked();

private:
        Ui::MainWindow *ui;
        QStringList fileNames;
        QString filePath;
        cv::Mat orginalPhoto;
        cv::Mat presentPhoto;
        cv::Mat savePhoto;
        cv::Mat houghTransformedPhoto;
        cv::Mat imageHistogram;
        cv::Mat saveHistogram;
        cv::Mat presentHist;
        int channels[3] = {0, 0, 0};
        std::pair<cv::Mat, cv::Mat> TwoThresholdImg;
//        cv::Mat grayPhoto;
//        cv::Mat colorPhoto;
//        cv::Mat BinaryPhoto;
};

#endif // MAINWINDOW_H
