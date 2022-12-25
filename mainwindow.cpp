#include "mainwindow.h"
#include "ui_mainwindow.h"

bool MainWindow::saveFlage = 0;
QString MainWindow::colorImgType = "";

void MainWindow::loadImageFromPath(QString path) {
    QPixmap pix;
    if(pix.load(path)) {
        ui->LB_ImagePresenter->setAlignment(Qt::AlignCenter);
        ui->LB_ImagePresenter->setPixmap(path);
        savePhoto = cv::imread(cv::samples::findFile(path.toStdString()), cv::IMREAD_ANYCOLOR); //, IMREAD_ANYCOLOR  cv::IMREAD_GRAYSCALE
        orginalPhoto = savePhoto.clone();
        imgTypeCheck(channels, savePhoto);
        loadHistogram(savePhoto);
        histogramPresenter();
    }
}

void MainWindow::photoPresenter(cv::Mat presentIt, QString currentType) {
    if (presentIt.channels() > 2 && (currentType == "BGR" || currentType == "bgr")) {
        cv::cvtColor(presentIt, presentPhoto, cv::COLOR_BGR2RGB);
        colorImgType = "RGB";
        //cv::Mat cvImageMap;
        //applyColorMap(savePhoto, cvImageMap, cv::COLORMAP_JET);
        //cv::cvtColor(cvImageMap, savePhoto, cv::COLOR_BGR2GRAY);
        // savePhoto = grayPhoto;
        QImage FinalCVImage = QImage((uchar*) presentPhoto.data, presentPhoto.cols, presentPhoto.rows, presentPhoto.step, QImage::Format_RGB888);
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        saveFlage = 1;

        imgTypeCheck(channels, savePhoto);
    }
    else if (presentIt.channels() < 2 && (currentType == "gray" || currentType == "GRAY")){
        //QImage FinalCVImage = QImage((uchar*) presentPhoto.data, presentPhoto.cols, presentPhoto.rows, presentPhoto.step, QImage::Format_Grayscale8);
        //ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(QImage((unsigned char*) presentIt.data, presentIt.cols, presentIt.rows, QImage::Format_Grayscale8)));
        colorImgType = "gray";
        saveFlage = 1;
        imgTypeCheck(channels, savePhoto);
    }
    else if (presentIt.channels() > 2 && (currentType == "gray" || currentType == "GRAY")) {
        cv::cvtColor(presentIt, presentPhoto, cv::COLOR_BGR2GRAY);
        QImage FinalCVImage = QImage((uchar*) presentPhoto.data, presentPhoto.cols, presentPhoto.rows, presentPhoto.step, QImage::Format_Grayscale8);
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        colorImgType = "gray";
        saveFlage = 1;
        imgTypeCheck(channels, savePhoto);
    }
}

void MainWindow::loadHistogram(cv::Mat src) {
    imageProcessing proc;
    imageHistogram = proc.generateHistogram(src, 0);
}

bool MainWindow::pathCheck(void) {
    if (!fileNames.isEmpty() && QString::compare(fileNames[0], QString()) == 0) {
        QMessageBox::warning(this, tr("Warning"), tr("please opean an image at first"), QMessageBox::Ok, QMessageBox::Ok);
        return 0;
    }
    else if (fileNames.isEmpty()) {
        QMessageBox::warning(this, tr("Warning"), tr("please opean an image at first"), QMessageBox::Ok, QMessageBox::Ok);
        return 0;
    }
    else return 1;
}

void MainWindow::imgTypeCheck(int channels[], cv::Mat srcImg) {
    channels[0] = srcImg.size().width;
    channels[1] = srcImg.size().height;
    channels[2] = srcImg.channels();
    if (srcImg.channels() == 3) {
        colorImgType = "BGR";
    }
    else colorImgType = "GRAY";

    QString barSts = QString("width:%1      height:%2      channels:%3").arg(QString::number(channels[0]), QString::number(channels[1]), QString::number(channels[2]));
    QByteArray ba = barSts.toLocal8Bit();
    const char *c_str = ba.data();
    ui->statusbar->showMessage(tr(c_str));
    // qDebug() << "width:" << channels[0] << " height:"<< channels[1] << " channels:" << channels[2] << "\n";
}

cv::Mat MainWindow::changeLayer(cv::Mat srcImg, QString currentType) {
    if (currentType == "RGB") {
        //cv::cvtColor(srcImg, srcImg, cv::COLOR_RGB2BGR);
        //colorImgType = "BGR";
        return srcImg;
    }
    else if (currentType == "BGR") {
        cv::cvtColor(srcImg, srcImg, cv::COLOR_BGR2RGB);
        colorImgType = "RGB";
        return srcImg;
    }
    return srcImg;
}

cv::Mat MainWindow::get_StructureElement_2d_odd() {
    QString kdata = ui->TE_convolveKernel->toPlainText();
    QStringList strList = kdata.split(QRegularExpression("[\n, ]"), Qt::SkipEmptyParts);
    int kernelSize = sqrt(strList.count());
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F);
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            QString temp = strList.first();
            strList.pop_front();
            kernel.at<float>(i,j) = temp.toFloat();
        }
    }
    return kernel;
}

void MainWindow::colorChange(cv::Mat &src, QString currentType, QString changeTo) {
    if ((currentType == "BGR" || currentType == "bgr") && (changeTo == "gray" || changeTo == "GRAY")) {
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    }
}

//cv::Mat MainWindow::loadFromQrc(QString qrc, int flag = cv::IMREAD_COLOR)
//{
//    //double tic = double(getTickCount());

//    QFile file(qrc);
//    cv::Mat m;
//    if(file.open(QIODevice::ReadOnly))
//    {
//        qint64 sz = file.size();
//        std::vector<uchar> buf(sz);
//        file.read((char*)buf.data(), sz);
//        m = cv::imdecode(buf, flag);
//    }

//    //double toc = (double(getTickCount()) - tic) * 1000.0 / getTickFrequency();
//    //qDebug() << "OpenCV loading time: " << toc;

//    return m;
//}


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    QScreen *screen = QGuiApplication::primaryScreen();
    QRect  screenGeometry = screen->geometry();
    [[maybe_unused]]int height = screenGeometry.height();
    [[maybe_unused]]int width = screenGeometry.width();
    ui->verticalLayout->setSizeConstraint(QLayout::SetFixedSize);
    // ui->LB_ImagePresenter->setGeometry(0, 0, width/3, height/2);
    ui->LB_ImagePresenter->setScaledContents(true);
    ui->LB_ImagePresenter->setMargin(8);
    ui->LB_ImagePresenter->setMaximumSize(width/2, height/2);
    ui->LB_ImagePresenter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->LB_ImagePresenter->setStyleSheet("border: 3px solid gray;");

    //ui->horizontalLayout_Hist->setSizeConstraint(QLayout::SetFixedSize);
    ui->LB_HistPresenter->setScaledContents(true);
    //ui->LB_HistPresenter->setMargin(0);
    ui->LB_HistPresenter->setMaximumSize(width/3, height/2);
    //ui->LB_HistPresenter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    //ui->LB_HistPresenter->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Ignored );
    ui->LB_HistPresenter->setStyleSheet("border: 3px solid gray;");
    //ui->LB_HistPresenter->adjustSize();

    ui->LE_SaveFileName->setPlaceholderText("Saved by this name");
    ui->LE_BinsNumbers->setText(QString::number(256));
    ui->LE_BinsNumbers->setPlaceholderText(QString::number(256));
    ui->LE_saltAndPepperNoiseNum->setPlaceholderText(QString::number(5000));
    ui->LE_gaussianNoiseNum->setPlaceholderText("M 34");
    ui->LE_gaussianNoiseSigma->setPlaceholderText("S 50");
    ui->LE_impulseNoiseNum->setPlaceholderText(QString::number(5000));
    ui->TE_convolveKernel->setPlaceholderText("a, b, c\nd, e, f\ng, h, i");
    ui->PTE_codingInput->setPlaceholderText("a, b, c, d, e, f, h, i");
    ui->BT_changeColorChanels->setStyleSheet("background-color:lightgray");
    ui->BT_1DConvolve->setToolTip("coding:input=XD, coding:output=YD");
    ui->PTE_codingInput->setPlainText("‫‪5‬‫‪5‬‬‫‪5‬‬‫‪5‬‬\n88\n4444\n‬‬‫‪22222222222222222222\n7777777\n44");

    QObject::connect(ui->LE_R_upperCB_thresholding, &QLineEdit::textChanged/*SIGNAL(textEdited(Qstring))*/, [this](const QString& val)->void{ui->HS_R_thresholding->setValue(val.toInt());});
    connect(ui->LE_G_upperCB_thresholding, &QLineEdit::textChanged, [this](const QString& val)->void{ui->HS_G_thresholding->setValue(val.toInt());});
    connect(ui->LE_B_upperCB_thresholding, &QLineEdit::textChanged, [this](const QString& val)->void{ui->HS_B_thresholding->setValue(val.toInt());});
    connect(ui->LE_KClustering, &QLineEdit::textChanged, [this](const QString& val)->void{ui->HS_threshold_Kmean_silder->setValue(val.toInt());});
    connect(ui->HS_R_thresholding, &QSlider::valueChanged, [this](const int& val)->void{ui->LE_R_upperCB_thresholding->setText(locale().toString(val));});
    connect(ui->HS_G_thresholding, &QSlider::valueChanged, [this](const int& val)->void{ui->LE_G_upperCB_thresholding->setText(locale().toString(val));});
    connect(ui->HS_B_thresholding, &QSlider::valueChanged, [this](const int& val)->void{ui->LE_B_upperCB_thresholding->setText(locale().toString(val));});
    connect(ui->HS_threshold_Kmean_silder, &QSlider::valueChanged, [this](const int& val)->void{ui->LE_KClustering->setText(locale().toString(val));});

    QObject::connect(ui->LE_R_upperCB_thresholding_2, &QLineEdit::textChanged/*SIGNAL(textEdited(Qstring))*/, [this](const QString& val)->void{ui->HS_R_thresholding_2->setValue(val.toInt());});
    connect(ui->LE_G_upperCB_thresholding_2, &QLineEdit::textChanged, [this](const QString& val)->void{ui->HS_G_thresholding_2->setValue(val.toInt());});
    connect(ui->LE_B_upperCB_thresholding_2, &QLineEdit::textChanged, [this](const QString& val)->void{ui->HS_B_thresholding_2->setValue(val.toInt());});
    connect(ui->HS_R_thresholding_2, &QSlider::valueChanged, [this](const int& val)->void{ui->LE_R_upperCB_thresholding_2->setText(locale().toString(val));});
    connect(ui->HS_G_thresholding_2, &QSlider::valueChanged, [this](const int& val)->void{ui->LE_G_upperCB_thresholding_2->setText(locale().toString(val));});
    connect(ui->HS_B_thresholding_2, &QSlider::valueChanged, [this](const int& val)->void{ui->LE_B_upperCB_thresholding_2->setText(locale().toString(val));});
//    const QString imagePath{"/home/muhammad/MyCodes/OpenCV/Images/Django.jpg"};
//    QPixmap pix;
//    /** to check wether load ok */
//    if(pix.load(imagePath)) {
//        // cv::Mat img_color = loadFromQrc(imagePath.toStdString(), cv::IMREAD_UNCHANGED);
//        cv::Mat img_color = cv::imread(imagePath.toStdString(), cv::IMREAD_UNCHANGED);
//        cv::Mat img_colorMap;
//        applyColorMap(img_color, img_colorMap, cv::COLORMAP_JET);
//        cv::cvtColor(img_colorMap, img_color, cv::COLOR_BGR2RGB);

//        QImage FinalImage = QImage((uchar*) img_color.data, img_color.cols, img_color.rows, img_color.step, QImage::Format_RGB888);

//        /** scale pixmap to fit in label'size and keep ratio of pixmap */
//        pix = pix.scaled(ui->LB_ImagePresenter->size(),Qt::KeepAspectRatio);
//        ui->LB_ImagePresenter->setAlignment(Qt::AlignCenter);
//        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalImage));
//    }
    ui->BT_FFTtransform->setVisible(false);
}


MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::ShowMessage(QString type, QString text) {
    QByteArray typeBA = type.toLocal8Bit();
    QByteArray textBA = text.toLocal8Bit();
    const char *type1 = typeBA.data();
    const char *text1 = textBA.data();
    QMessageBox::warning(this, tr(type1), tr(text1), QMessageBox::Ok, QMessageBox::Ok);
}

void MainWindow::on_BT_Orginal_clicked() {
    if(pathCheck()) {
        loadImageFromPath(fileNames[0]);
    }
}


void MainWindow::on_BT_Gray_clicked() {
//    if (pathCheck()) {
//        orginalPhoto = cv::imread(fileNames[0].toStdString(), cv::IMREAD_UNCHANGED);
//        if (orginalPhoto.channels() > 2) {
//            cv::cvtColor(orginalPhoto, savePhoto, cv::COLOR_BGR2GRAY);
//            colorImgType = "GRAY";
//            cv::Mat cvImageMap;
//            //applyColorMap(savePhoto, cvImageMap, cv::COLORMAP_JET);
//            //cv::cvtColor(cvImageMap, savePhoto, cv::COLOR_BGR2GRAY);
//            // savePhoto = grayPhoto;
//            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8);
//            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
//            saveFlage = 1;

//            imgTypeCheck(channels, savePhoto);
//        }
//    }
//    else return;
    if (savePhoto.channels() > 2) {
        colorChange(savePhoto, "BGR", "GRAY");
        imgTypeCheck(channels, savePhoto);
        photoPresenter(savePhoto, colorImgType);
    }
}


void MainWindow::on_BT_Color_clicked() {
    if (pathCheck()) {
        orginalPhoto = cv::imread(fileNames[0].toStdString(), cv::IMREAD_UNCHANGED);
        cv::cvtColor(orginalPhoto, savePhoto, cv::COLOR_BGR2RGB);
        colorImgType = "RGB";
        cv::Mat cvImageMap;
        applyColorMap(savePhoto, cvImageMap, cv::COLORMAP_JET);
        cv::cvtColor(cvImageMap, savePhoto, cv::COLOR_BGR2RGB);
        colorImgType = "RGB";
        // savePhoto = colorPhoto;
        QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        saveFlage = 1;

        imgTypeCheck(channels, savePhoto);
    }
    else return;
}


void MainWindow::on_BT_Binary_clicked() {
//    filePath = QFileDialog::getOpenFileName(this, tr("chose"), tr("Images (*.png *.jpeg *.jpg *.tif)"));
//    if (QString::compare(filePath, QString()) != 0) {
//        QImage image;
//        bool valid = image.load(filePath);
//        if (valid) {
//            image = image.scaledToWidth(ui->LB_ImagePresenter->width(), Qt::SmoothTransformation);
//            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(image));
//        }
//    }
    if (pathCheck()) {
        orginalPhoto = cv::imread(fileNames[0].toStdString(), cv::IMREAD_GRAYSCALE);
        QVector<double> thresh{128.0, 100.0, 77.0};
        cv::threshold(orginalPhoto, savePhoto, thresh[2], 255.0, cv::THRESH_BINARY); //  | cv::THRESH_OTSU
        //cv::imshow("ok", grayPhoto);
        //cv::cvtColor(orginalPhoto, grayPhoto, cv::COLOR_BGR2GRAY);
        cv::Mat cvImageMap;
        applyColorMap(savePhoto, cvImageMap, cv::COLORMAP_BONE);
        cv::cvtColor(cvImageMap, savePhoto, cv::COLOR_BGR2GRAY);
        colorImgType = "GRAY";
        // savePhoto = grayPhoto;
        // BinaryPhoto = grayPhoto;
        QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8); //Indexed8
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        saveFlage = 1;

        imgTypeCheck(channels, savePhoto);
    }
}


void MainWindow::on_BT_OpenPath_clicked() {
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("Images (*.png *.jpeg *.jpg *.tif)"));
    dialog.setViewMode(QFileDialog::Detail);
    if (dialog.exec()) fileNames = dialog.selectedFiles();
    else return;
    // QString filePath = QFileDialog::getExistingDirectory(this, "Choose Folder");
    // QFileDialog::getExistingDirectory(this,"Choose Folder");
    // if (filePath.isEmpty()) return;
    ui->LE_PathPresenter->setText(fileNames[0]);
    if(fileNames[0].isEmpty() || fileNames[0].isNull()) return;
    else this->loadImageFromPath(fileNames[0]);
}


void MainWindow::on_BT_Save_clicked() {
    if (saveFlage == 1) {
        QStringList newPathName;
        QStringList fileFormat;
        QString savePath;
        newPathName.append(fileNames[0].split("/"));
        fileFormat = newPathName.last().split(".");
        newPathName.pop_back();
        if (!ui->LE_SaveFileName->text().isEmpty()) {
            newPathName.append(ui->LE_SaveFileName->text() + "." + fileFormat.last());
        }
        else {
            newPathName.append(QString::number(QRandomGenerator::global()->generate()) + "." + fileFormat.last());
        }
        savePath = newPathName.join("/");
        bool saveCheck;
        // std::generate(1, 999, []() { return QString::number(QRandomGenerator::global()->generate64()); });
        saveCheck = cv::imwrite(savePath.toStdString(), changeLayer(savePhoto, "BGR"));
        if (saveCheck == false) {
            QMessageBox::warning(this, tr("Warning"), tr("Mission - Saving the image, FAILED"), QMessageBox::Ok, QMessageBox::Ok);
        }
        else saveFlage = 0;
    }
    else if (pathCheck()){
        //QMessageBox::warning(this, tr("Warning"), tr("Please opean an image at first"), QMessageBox::Ok, QMessageBox::Ok);
        return;
    }
    else {
        //QMessageBox::warning(this, tr("Warning"), tr("Image dont changed"), QMessageBox::Ok, QMessageBox::Ok);
    }
}


void MainWindow::on_BT_houghTransform_clicked() {
    if (pathCheck()) {
        imageProcessing imgProc;
        houghTransformedPhoto = imgProc.cvHoughTransform(savePhoto).first;

        cv::cvtColor(houghTransformedPhoto, savePhoto, cv::COLOR_BGR2GRAY);
        colorImgType = "GRAY";
        cv::Mat cvImageMap;
        applyColorMap(savePhoto, cvImageMap, cv::COLORMAP_JET);
        cv::cvtColor(cvImageMap, savePhoto, cv::COLOR_BGR2GRAY);
        colorImgType = "GRAY";
        // savePhoto = houghTransformedPhoto;
        // savePhoto = BinaryPhoto;
        QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8);
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));

        saveFlage = 1;
    }
}


void MainWindow::on_BT_probabilisticHoughTransform_clicked() {
    if (pathCheck()) {
        imageProcessing imgProc;
        houghTransformedPhoto = imgProc.cvHoughTransform(savePhoto).second;
        cv::cvtColor(houghTransformedPhoto, savePhoto, cv::COLOR_BGR2GRAY);
        colorImgType = "GRAY";
        cv::Mat cvImageMap;
        applyColorMap(savePhoto, cvImageMap, cv::COLORMAP_JET);
        cv::cvtColor(cvImageMap, savePhoto, cv::COLOR_BGR2GRAY);
        colorImgType = "GRAY";
        // savePhoto = houghTransformedPhoto;
        // savePhoto = BinaryPhoto;
        QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8);
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        saveFlage = 1;
    }
}


void MainWindow::on_BT_MyHoughTransform_clicked() {
//    QStringList arguments { "imgprocessing.py" };
//    QProcess p;
//    p.start("python", arguments);
//    QString p_stdout = p.readAll();
//    p.waitForFinished();

    if(pathCheck()) {
        QString  program( "python3" );
        QStringList  args = QStringList() << "../CV_test/imgprocessing.py" << fileNames[0];
        int exitCode = QProcess::execute( program, args );
        qDebug() << exitCode;
    }

//    if(pathCheck()) {
//        imageProcessing imgProc;
//        houghTransformedPhoto = imgProc.myHoughTransform(savePhoto);
//        cv::cvtColor(houghTransformedPhoto, savePhoto, cv::COLOR_BGR2GRAY);
//        cv::Mat cvImageMap;
//        applyColorMap(savePhoto, cvImageMap, cv::COLORMAP_JET);
//        cv::cvtColor(cvImageMap, savePhoto, cv::COLOR_BGR2GRAY);
//        // savePhoto = houghTransformedPhoto;
//        // savePhoto = BinaryPhoto;
//        QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8);
//        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
//        saveFlage = 1;
//    }
}

// #############################################################################
//                              Histogram
// #############################################################################

void MainWindow::on_BT_GenenrateHistogram_clicked() {
    imageProcessing imgProc;
    if(pathCheck()) {
        QString  program( "python3" );
        if (ui->LE_BinsNumbers->text().isEmpty() && ui->LE_BinsNumbers->text().toInt() < 1) {
            ui->LE_BinsNumbers->setText(QString::number(256));
        }
        QStringList  args = QStringList() << "../CV_test/clacHist2d.py" << fileNames[0] << ui->LE_BinsNumbers->text();
        int exitCode = QProcess::execute( program, args );
        qDebug() << exitCode;
        if(imageHistogram.channels() < 2) {
            imageHistogram = imgProc.generateHistogram(savePhoto, 1);
        }
    }


//    cv::cvtColor(imageHistogram, saveHistogram, cv::COLOR_BGR2RGB);
//    cv::Mat cvImageMap;
//    applyColorMap(saveHistogram, cvImageMap, cv::COLORMAP_JET);
//    cv::cvtColor(cvImageMap, saveHistogram, cv::COLOR_BGR2RGB);
//    // savePhoto = colorPhoto;
//    QImage FinalCVImage = QImage((uchar*) saveHistogram.data, saveHistogram.cols, saveHistogram.rows, saveHistogram.step, QImage::Format_RGB888);
//    ui->LB_HistPresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
}

void MainWindow::histogramPresenter() {
    if(imageHistogram.channels() < 2) {
//        for(size_t i = 0; i < 256; i++) {
//            qDebug()<< imageHistogram.at<float>(i) << "\n";
//        }
        cv::cvtColor(imageHistogram, saveHistogram, cv::COLOR_GRAY2RGB);
        colorImgType = "RGB";
        cv::Mat cvImageMap;
        //applyColorMap(saveHistogram, cvImageMap, cv::COLORMAP_BONE);
        // cv::cvtColor(cvImageMap, saveHistogram, cv::COLOR_BGR2GRAY);
        // savePhoto = colorPhoto;
        QImage FinalCVImage = QImage((uchar*) saveHistogram.data, saveHistogram.cols, saveHistogram.rows, saveHistogram.step, QImage::Format_Grayscale8);
        ui->LB_HistPresenter->setPixmap(QPixmap::fromImage(FinalCVImage).scaled(ui->LB_HistPresenter->size(), Qt::IgnoreAspectRatio /*Qt::SmoothTransformation*/));
    }
    else if (imageHistogram.channels() == 3) {
        cv::cvtColor(imageHistogram, saveHistogram, cv::COLOR_BGR2RGB);
        colorImgType = "RGB";
        cv::Mat cvImageMap;
        //applyColorMap(saveHistogram, cvImageMap, cv::COLORMAP_TWILIGHT); //COLORMAP_PINK  COLORMAP_DEEPGREEN
        //cv::cvtColor(cvImageMap, saveHistogram, cv::COLOR_BGR2RGB);
        // savePhoto = colorPhoto;
        QImage FinalCVImage = QImage((uchar*) saveHistogram.data, saveHistogram.cols, saveHistogram.rows, saveHistogram.step, QImage::Format_RGB888);
        ui->LB_HistPresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
    }
}

void MainWindow::on_BT_calcHist3d_clicked() {
    if(pathCheck()) {
        QString  program( "python3" );
        if (ui->LE_BinsNumbers->text().isEmpty() && ui->LE_BinsNumbers->text().toInt() < 1) {
            ui->LE_BinsNumbers->setText(QString::number(256));
        }
        QStringList  args = QStringList() << "../CV_test/CalcHist.py" << fileNames[0] << ui->LE_BinsNumbers->text();
        int exitCode = QProcess::execute( program, args );
        qDebug() << exitCode;
    }
}

void MainWindow::on_BT_CVHistogram_clicked() {
    imageProcessing imgProc;
    imageHistogram = imgProc.generateHistogram(savePhoto, 1);
}

void MainWindow::on_BT_equalizeHistogram_clicked() {
    imageProcessing prc;
    savePhoto = prc.HistogramEqualizer(savePhoto);
    cv::cvtColor(savePhoto, presentHist, cv::COLOR_BGR2RGB);
    colorImgType = "RGB";
    QImage FinalCVImage = QImage((uchar*) presentHist.data, presentHist.cols, presentHist.rows, presentHist.step, QImage::Format_RGB888);
    ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
    saveFlage = 1;
    loadHistogram(savePhoto);
    histogramPresenter();
}

// #############################################################################
//
// #############################################################################

void MainWindow::on_BT_AddNoise_clicked() {
    if (ui->RB_saltAndPepperNoise->isChecked()) {
        imageProcessing prc;
        savePhoto = prc.noise_saltAndPepper(savePhoto, ui->LE_saltAndPepperNoiseNum->text().toInt());
        if (savePhoto.channels() < 2) {
            cv::cvtColor(savePhoto, savePhoto, cv::COLOR_GRAY2RGB);
            colorImgType = "RGB";
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
        else if (savePhoto.channels() == 3) {
//            auto o = cv::FastFeatureDetector::getType(savePhoto);
//            if (savePhoto.type() == cv::FastFeatureDetector::getType())
            cv::cvtColor(savePhoto, savePhoto, cv::COLOR_BGR2RGB);
            colorImgType = "RGB";
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
    }
    else if (ui->RB_gaussianNoise->isChecked()) {
        imageProcessing prc;
        savePhoto = prc.noise_gaussian(savePhoto, ui->LE_gaussianNoiseNum->text().toFloat(), ui->LE_gaussianNoiseSigma->text().toFloat());
        if (savePhoto.channels() < 2) {
            cv::cvtColor(savePhoto, savePhoto, cv::COLOR_GRAY2RGB);
            colorImgType = "RGB";
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
        else if (savePhoto.channels() == 3) {
            cv::cvtColor(savePhoto, savePhoto, cv::COLOR_BGR2RGB);
            colorImgType = "RGB";
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
    }
    else if (ui->RB_impulseNoise->isChecked()) {
        imageProcessing prc;
        savePhoto = prc.noise_impulse(savePhoto, ui->LE_impulseNoiseNum->text().toInt());
        if (savePhoto.channels() < 2) {
            cv::cvtColor(savePhoto, savePhoto, cv::COLOR_GRAY2RGB);
            colorImgType = "RGB";
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
        else if (savePhoto.channels() == 3) {
            cv::cvtColor(savePhoto, savePhoto, cv::COLOR_BGR2RGB);
            colorImgType = "RGB";
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
    }
}

// #############################################################################
//                              Convolution
// #############################################################################

void MainWindow::on_BT_convolve2d_clicked() {
    QString kdata = ui->TE_convolveKernel->toPlainText();
    QStringList strList = kdata.split(QRegularExpression("[\n, ]"), Qt::SkipEmptyParts);
    int kernelSize = sqrt(strList.count());
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F);
//    cv::Mat kernel = (cv::Mat_<double>(sqrt(kernelSize),sqrt(kernelSize)) << 0, 0, 0, 0, 1, 0, 0, 0, 0);
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            QString temp = strList.first();
            strList.pop_front();
            kernel.at<float>(i,j) = temp.toFloat();
        }
    }
    //normalize kernel so that sum of all the elements is equal to 1
//    kernel = kernel / (float)(kernelSize * kernelSize);
    imageProcessing proc;
    savePhoto = proc.convolution2d(savePhoto, kernel);
    if (savePhoto.channels() < 2) {
        cv::cvtColor(savePhoto, savePhoto, cv::COLOR_GRAY2RGB);
        colorImgType = "RGB";
        QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8);
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        saveFlage = 1;
    }
    else if (savePhoto.channels() == 3) {
        cv::cvtColor(savePhoto, savePhoto, cv::COLOR_BGR2RGB);
        colorImgType = "RGB";
        QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        saveFlage = 1;
    }
    saveFlage = 1;
    //qDebug() << kernelSize;
}

void MainWindow::on_BT_1DConvolve_clicked() {
    if (ui->PTE_codingInput->document()->isEmpty()) {
        ShowMessage("Warning", "use coding->input as XD and coding->output as YD");
        return;
    }
    QString XmaskData = ui->PTE_codingInput->toPlainText();
    QString YmaskData = ui->PTE_codingOutput->toPlainText();
    QStringList XstrList = XmaskData.split(QRegularExpression("[\n, ]"), Qt::SkipEmptyParts);
    QStringList YstrList = YmaskData.split(QRegularExpression("[\n, ]"), Qt::SkipEmptyParts);
    int XkernelSize = XstrList.count();
    int YkernelSize = YstrList.count();
//    cv::Mat Xkernel = cv::Mat::ones(XkernelSize, XkernelSize, CV_32F);
//    cv::Mat Ykernel = cv::Mat::ones(YkernelSize, YkernelSize, CV_32F);
//    for (int i = 0; i < XkernelSize; i++) {
//        for (int j = 0; j < XkernelSize; j++) {
//            QString temp = XstrList.first();
//            XstrList.pop_front();
//            Xkernel.at<float>(i,j) = temp.toFloat();
//        }
//    }
//    for (int i = 0; i < YkernelSize; i++) {
//        for (int j = 0; j < YkernelSize; j++) {
//            QString temp = YstrList.first();
//            YstrList.pop_front();
//            Ykernel.at<float>(i,j) = temp.toFloat();
//        }
//    }
    cv::Mat KX(1,XkernelSize,CV_64F); // 10 doubles in a single row
    for (int i=0; i<XkernelSize; i++) {
        QString temp = XstrList.first();
        XstrList.pop_front();
        KX.at<double>(0,i) = temp.toFloat(); // set the column in row 0
    }
    cv::Mat KY(1,YkernelSize,CV_64F); // 10 doubles in a single row
    for (int i=0; i<YkernelSize; i++) {
        QString temp = YstrList.first();
        YstrList.pop_front();
        KY.at<double>(0,i) = temp.toFloat(); // set the column in row 0
    }
    imageProcessing proc;
    savePhoto = proc.sepConvolution2d(savePhoto, KX, KY);
    if (savePhoto.channels() < 2) {
        //cv::cvtColor(savePhoto, savePhoto, cv::COLOR_GRAY2RGB);
        //colorImgType = "RGB";
        QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_Grayscale8);
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        saveFlage = 1;
    }
    else if (savePhoto.channels() == 3) {
        cv::cvtColor(savePhoto, savePhoto, cv::COLOR_BGR2RGB);
        colorImgType = "RGB";
        QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
        ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
        saveFlage = 1;
    }
    saveFlage = 1;
}

// #############################################################################
//                              Noise Filter
// #############################################################################

void MainWindow::on_BT_Filter_clicked() {
    if(ui->RB_MeanFilter->isChecked()) {
        if (ui->RB_3SizeFilter->isChecked()) {
            imageProcessing prc;
            savePhoto = prc.NormalizedBlockFilter(savePhoto, 3);
            savePhoto = changeLayer(savePhoto, colorImgType);
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
        else if (ui->RB_5SizeFilter->isChecked()) {
            imageProcessing prc;
            savePhoto = prc.NormalizedBlockFilter(savePhoto, 5);
            savePhoto = changeLayer(savePhoto, colorImgType);
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
        else if (ui->RB_7SizeFilter->isChecked()) {
            imageProcessing prc;
            savePhoto = prc.NormalizedBlockFilter(savePhoto, 7);
            savePhoto = changeLayer(savePhoto, colorImgType);
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
    }
    else if (ui->RB_GaussianFilter->isChecked()) {
        if (ui->RB_3SizeFilter->isChecked()) {
            imageProcessing prc;
            savePhoto = prc.gaussianFilter(savePhoto, 3);
            savePhoto = changeLayer(savePhoto, colorImgType);
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
        else if (ui->RB_5SizeFilter->isChecked()) {
            imageProcessing prc;
            savePhoto = prc.gaussianFilter(savePhoto, 5);
            savePhoto = changeLayer(savePhoto, colorImgType);
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
        else if (ui->RB_7SizeFilter->isChecked()) {
            imageProcessing prc;
            savePhoto = prc.gaussianFilter(savePhoto, 7);
            savePhoto = changeLayer(savePhoto, colorImgType);
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
    }
    else if (ui->RB_MedianFilter->isChecked()) {
        if (ui->RB_3SizeFilter->isChecked()) {
            imageProcessing prc;
            savePhoto = prc.medianFilter(savePhoto, 3);
            savePhoto = changeLayer(savePhoto, colorImgType);
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
        else if (ui->RB_5SizeFilter->isChecked()) {
            imageProcessing prc;
            savePhoto = prc.medianFilter(savePhoto, 5);
            savePhoto = changeLayer(savePhoto, colorImgType);
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
        else if (ui->RB_7SizeFilter->isChecked()) {
            imageProcessing prc;
            savePhoto = prc.medianFilter(savePhoto, 7);
            savePhoto = changeLayer(savePhoto, colorImgType);
            QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
            ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));
            saveFlage = 1;
        }
    }
}

// #############################################################################
//                              compression
// #############################################################################

//std::vector<...> stdVec;
//QVector<...> qVec = QVector<...>(stdVec.begin(), stdVec.end());
//std::vector<T> stdVec;
//QVector<T> qVec = QVector<T>::fromStdVector(stdVec);
void MainWindow::on_BT_huffmanCoding_clicked() {
    QString cData;
    cData = ui->PTE_codingInput->toPlainText();
    QStringList elmList = cData.split(QRegularExpression("[\n, ]"), Qt::SkipEmptyParts);
    QString firstListElm;
    QVector<int> elementCount;
    QVector<QString> elementName;
    int numberOfElm;
    while (!elmList.isEmpty()) {
        firstListElm = elmList.first();
        numberOfElm = elmList.count(firstListElm);
        elementName.push_back(firstListElm);
        elementCount.push_back(numberOfElm);
        elmList.removeAll(firstListElm);
    }
    qDebug() << elementName;
    qDebug() << elementCount;
    // elmList.removeDuplicates();
}

// #############################################################################
//                       Segmentation - Thresholding
// #############################################################################

void MainWindow::on_BT_ThresholdingSegmentation_clicked() {
    int rl = ui->LE_R_lowerCB_thresholding->text().toInt();
    int gl = ui->LE_G_lowerCB_thresholding->text().toInt();
    int bl = ui->LE_B_lowerCB_thresholding->text().toInt();
    int ru = ui->LE_R_upperCB_thresholding->text().toInt();
    int gu = ui->LE_G_upperCB_thresholding->text().toInt();
    int bu = ui->LE_B_upperCB_thresholding->text().toInt();

    imageProcessing prc;
    savePhoto = prc.inrange_Segmentaion(savePhoto, rl, gl, bl, ru, gu, bu);
    photoPresenter(savePhoto, "BGR");
    TwoThresholdImg.first = savePhoto;
    saveFlage = 1;
}

void MainWindow::on_BT_ThresholdingSegmentation_2_clicked() {
    int rl2 = ui->LE_R_lowerCB_thresholding_2->text().toInt();
    int gl2 = ui->LE_G_lowerCB_thresholding_2->text().toInt();
    int bl2 = ui->LE_B_lowerCB_thresholding_2->text().toInt();
    int ru2 = ui->LE_R_upperCB_thresholding_2->text().toInt();
    int gu2 = ui->LE_G_upperCB_thresholding_2->text().toInt();
    int bu2 = ui->LE_B_upperCB_thresholding_2->text().toInt();

    imageProcessing prc;
    savePhoto = prc.inrange_Segmentaion(savePhoto, rl2, gl2, bl2, ru2, gu2, bu2);
    photoPresenter(savePhoto, "BGR");
    TwoThresholdImg.second = savePhoto;
    saveFlage = 1;
}

void MainWindow::on_BT_ThrSeg_orgAndT1_clicked() {
    cv::Mat orgAndT1;
    cv::hconcat(orginalPhoto, TwoThresholdImg.first, orgAndT1);
    photoPresenter(orgAndT1, "BGR");
    saveFlage = 1;
}

void MainWindow::on_BT_ThrSeg_orgAndT2_clicked() {
    cv::Mat orgAndT2;
    cv::hconcat(orginalPhoto, TwoThresholdImg.second, orgAndT2);
    photoPresenter(orgAndT2, "BGR");
    saveFlage = 1;
}

void MainWindow::on_BT_ThrSeg_T1AndT2_clicked() {
    cv::Mat T1AndT2;
    cv::hconcat(TwoThresholdImg.first, TwoThresholdImg.second, T1AndT2);
    photoPresenter(T1AndT2, "BGR");
    saveFlage = 1;
}

void MainWindow::on_BT_labeling_morphology_clicked() {
    imageProcessing prc;
    savePhoto = prc.labeling_Morphology(savePhoto);
    imgTypeCheck(channels, savePhoto);
    photoPresenter(savePhoto, colorImgType);
    saveFlage = 1;
}

// #############################################################################
//                       Segmentation - Clustering
// #############################################################################

void MainWindow::on_BT_ClusteringSegmentation_clicked() {
    int k = ui->LE_KClustering->text().toInt();
    imageProcessing prc;
    savePhoto = prc.kmean_Segmentaion(savePhoto, k, savePhoto.channels());
    imgTypeCheck(channels, savePhoto);
    photoPresenter(savePhoto, colorImgType);
    saveFlage = 1;
}

// #############################################################################
//                      Morphological Transformations
// #############################################################################

void MainWindow::on_BT_erosion_Morphology_clicked() {
    imageProcessing prc;
    savePhoto = prc.Erosion_Morphology(savePhoto, get_StructureElement_2d_odd());
    imgTypeCheck(channels, savePhoto);
    photoPresenter(savePhoto, colorImgType);
    saveFlage = 1;
}

void MainWindow::on_BT_dilation_Morphology_clicked() {
    imageProcessing prc;
    savePhoto = prc.dilation_Morphology(savePhoto, get_StructureElement_2d_odd());
    imgTypeCheck(channels, savePhoto);
    photoPresenter(savePhoto, colorImgType);
    saveFlage = 1;
}

void MainWindow::on_BT_opening_Morphology_clicked() {
    imageProcessing prc;
    savePhoto = prc.opening_Morphology(savePhoto, get_StructureElement_2d_odd());
    imgTypeCheck(channels, savePhoto);
    photoPresenter(savePhoto, colorImgType);
    saveFlage = 1;
}

void MainWindow::on_BT_closing_Morphology_clicked() {
    imageProcessing prc;
    savePhoto = prc.closing_Morphology(savePhoto, get_StructureElement_2d_odd());
    imgTypeCheck(channels, savePhoto);
    photoPresenter(savePhoto, colorImgType);
    saveFlage = 1;
}

// #############################################################################
//             furier transform - lowpass and highpass filter
// #############################################################################

void MainWindow::on_BT_DFTtransform_clicked() {
//    int radius=35;
//    cv::Mat img, complexImg, filter, filterOutput, imgOutput, planes[2];
//    img = cv::imread(fileNames[0].toStdString(), 0);
//    if(img.empty())
//    {
//        qDebug() << "empty source";
//    }
//    imageProcessing prc;

//    complexImg = prc.computeDFT(img);
//    filter = complexImg.clone();

//    prc.lowpassFilter(filter, radius); // create an ideal low pass filter

//    prc.fftShift(complexImg); // rearrage quadrants
//    mulSpectrums(complexImg, filter, complexImg, 0); // multiply 2 spectrums
//    prc.fftShift(complexImg); // rearrage quadrants

//    // compute inverse
//    idft(complexImg, complexImg);

//    split(complexImg, planes);
//    cv::normalize(planes[0], imgOutput, 0, 1, cv::NORM_MINMAX);  // CV_MINMAX

//    split(filter, planes);
//    normalize(planes[1], filterOutput, 0, 1, cv::NORM_MINMAX);  // CV_MINMAX

//    savePhoto = imgOutput.clone();
//    imgTypeCheck(channels, savePhoto);
//    photoPresenter(imgOutput, "gray");

//    imshow("Input image", img);
//    waitKey(0);
//    imshow("Filter", filterOutput);
//    waitKey(0);
//    imshow("Low pass filter", imgOutput);
//    waitKey(0);
//    destroyAllWindows();
    if(pathCheck()) {
        QString  program( "c++" );
        QStringList  args = QStringList() << "../CV_test/lowpass" << fileNames[0];
        int exitCode = QProcess::execute( program, args );
        qDebug() << exitCode;
    }
}

// #############################################################################
//                              Edge Detector
// #############################################################################

void MainWindow::on_BT_SobelEdgeDetector_clicked() {
    imageProcessing prc;
    if (ui->RB_Xaxis_sobel->isChecked()) {
        savePhoto = prc.Sobel_EdgeDetector(savePhoto, "x");
        imgTypeCheck(channels, savePhoto);
        photoPresenter(savePhoto, colorImgType);
        saveFlage = 1;
    }
    else if (ui->RB_Yaxis_sobel->isChecked()) {
        savePhoto = prc.Sobel_EdgeDetector(savePhoto, "y");
        imgTypeCheck(channels, savePhoto);
        photoPresenter(savePhoto, colorImgType);
        saveFlage = 1;
    }
    else if (ui->RB_XYaxis_sobel->isChecked()) {
        savePhoto = prc.Sobel_EdgeDetector(savePhoto, "xy");
        imgTypeCheck(channels, savePhoto);
        photoPresenter(savePhoto, colorImgType);
        saveFlage = 1;
    }

}

void MainWindow::on_BT_CannyEdgeDetector_clicked() {

}

void MainWindow::on_BT_PerwittEdgeDetector_clicked() {

}

// #############################################################################
//
// #############################################################################

void MainWindow::on_BT_changeColorChanels_clicked() {
    cv::cvtColor(savePhoto, savePhoto, cv::COLOR_BGR2RGB);
    QImage FinalCVImage = QImage((uchar*) savePhoto.data, savePhoto.cols, savePhoto.rows, savePhoto.step, QImage::Format_RGB888);
    ui->LB_ImagePresenter->setPixmap(QPixmap::fromImage(FinalCVImage));

}

