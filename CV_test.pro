QT       += core gui
#QT += charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets #printsupport

CONFIG += c++14

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    imageprocessing.cpp \
    main.cpp \
    mainwindow.cpp


# ../QCustomPlot/qcustomplot/qcustomplot.cpp \

HEADERS += \
    imageprocessing.h \
    mainwindow.h



# ../QCustomPlot/qcustomplot/qcustomplot.h \

FORMS += \
    mainwindow.ui

QMAKE_LFLAGS += -no-pie

#QT_QPA_PLATFORM=wayland

# OpenCV
#unix {
#    CONFIG += link_pkgconfig
#    PKGCONFIG += opencv4
#}
INCLUDEPATH += /usr/local/include/opencv4 #/usr/local/include/opencv4
LIBS += $(shell pkg-config opencv --libs)
LIBS += -L/usr/local/lib -lopencv_gapi -lopencv_stitching -lopencv_aruco -lopencv_barcode -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform -lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_highgui -lopencv_datasets -lopencv_text -lopencv_plot -lopencv_videostab -lopencv_videoio -lopencv_wechat_qrcode -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_video -lopencv_dnn -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core# -nostartfiles -shared
#LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -nostdlib  #-lstdc++ -fstack-protector -lopencv_videoio -lopencv_objdetect -lopencv_imgcodecs
#LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
## ==================== open cv ==========================
#INCLUDEPATH += D:\opencv\build\include
#LIBS += D:\opencv-build\bin\libopencv_core320.dll
#LIBS += D:\opencv-build\bin\libopencv_highgui320.dll
#LIBS += D:\opencv-build\bin\libopencv_imgcodecs320.dll
#LIBS += D:\opencv-build\bin\libopencv_imgproc320.dll
#LIBS += D:\opencv-build\bin\libopencv_features2d320.dll
#LIBS += D:\opencv-build\bin\libopencv_calib3d320.dll
## more correct variant, how set includepath and libs for mingw
## add system variable: OPENCV_SDK_DIR=D:/opencv/opencv-build/install
## read http://doc.qt.io/qt-5/qmake-variable-reference.html#libs
##INCLUDEPATH += $$(OPENCV_SDK_DIR)/include
##LIBS += -L$$(OPENCV_SDK_DIR)/x86/mingw/lib \
##        -lopencv_core320        \
##        -lopencv_highgui320     \
##        -lopencv_imgcodecs320   \
##        -lopencv_imgproc320     \
##        -lopencv_features2d320  \
##        -lopencv_calib3d320
## =======================================================

# ICON = Camera-icon.png

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    CalcHist.py \
    clacHist2d.py \
    imgprocessing.py
