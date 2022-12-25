#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    QScreen *screen = QGuiApplication::primaryScreen();
    QRect  screenGeometry = screen->geometry();
    [[maybe_unused]]int height = screenGeometry.height();
    [[maybe_unused]]int width = screenGeometry.width();

    // w.setGeometry(100, 100, width, height); // ,,680, 600 or width*0.6, height*0.8
    w.setWindowState(Qt::WindowMaximized);
    w.show();
    return a.exec();
}
