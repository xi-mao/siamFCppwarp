#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include<testpt.h>
#include<siamfcpp.h>
int main(int argc, char *argv[])
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    const QUrl url(QStringLiteral("qrc:/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);


    testpt testp;

    testp.testpt_load("siamfcpp_features_cpu.pt",torch::DeviceType::CPU);
  // testp.testpt_loadtrack("siamfcpp_track_cpu.pt",torch::DeviceType::CPU);

//siamfcpp siampp("siamfcpp.pt",torch::DeviceType::CPU);

 engine.load(url);
    return app.exec();
}
