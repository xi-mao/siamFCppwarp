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
   //   cv::Mat image = cv::imread("00001.jpg");
   // cv::Rect roi=   cv::selectROI("nn",image);
  //  testp.ini(image,roi, torch::DeviceType::CPU);


 cv::VideoCapture   vs = cv::VideoCapture("../video/bag.avi");

  bool  frst = true;
  cv::Mat frame ;
        // ;
    while ( vs.read(frame)){





            if (frst) {
               // cv::Rect box = cv::selectROI("window_name",
                 //                   frame

                //                    );
               cv::Rect box(313,135,109,123);
                cv::putText(frame,
                            "", cv::Point(128, 20),
                            cv::FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
                            1);
                cv::rectangle(frame, box,
                              cv::Scalar(0, 255, 0));

                cv::imshow("window_name", frame);
                cv::waitKey(1);
               std::cout<<"Roi  "<<box<<std::endl;
               cv::imwrite("ini.jpg",frame);
                testp.ini(frame, box, torch::DeviceType::CPU);

                frst= false;
            }
            else{
            cv::Rect roi=  testp.update(frame);

               cv::Mat show_frame = frame.clone();

               // bbox_pred = xywh2xyxy(rect_pred)
               // bbox_pred = tuple(map(int, bbox_pred))

                cv::putText(show_frame,
                            "", cv::Point(128, 20),
                            cv::FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
                            1);

                 std::cout<<"roi  "<<roi<<std::endl;
                cv::rectangle(show_frame, roi ,
                              cv::Scalar(0, 255, 0));



                cv::imshow("window_name2", show_frame);
                cv::waitKey(1);
            }
    }


 engine.load(url);
    return app.exec();
}
