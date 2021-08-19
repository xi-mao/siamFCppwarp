#include <QGuiApplication>
#include <QQmlApplicationEngine>
//#include<testpt.h>
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


  //  testpt testp(nullptr,torch::DeviceType::CUDA);
siamfcpp  siamfcpp(nullptr,torch::DeviceType::CUDA);


 cv::VideoCapture   vs = cv::VideoCapture("../video/video.avi");

  bool  frst = true;
  cv::Mat frame ;
        // ;

  float fps_c=0;
    int c=0;
    while ( vs.read(frame)){


c++;


            if (frst) {
                cv::Rect box = cv::selectROI("window_name",
                                    frame

                                    );
             //  cv::Rect box(313,135,109,123);







                siamfcpp.ini(frame, box);

                frst= false;

            }
            else{

                 int start =cv::getTickCount();

                cv::Rect roi=  siamfcpp.update(frame);
                int  end =cv::getTickCount();
               cv::Mat show_frame = frame.clone();
                float cout=1/((end-start)/cv::getTickFrequency());
                fps_c+=cout;


                cv::putText(show_frame,
                           "FPS "+ std::to_string( cout), cv::Point(128, 20),
                            cv::FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
                           1);


                cv::rectangle(show_frame, roi ,
                              cv::Scalar(0, 255,0),3);



                cv::imshow("window_name2", show_frame);
                cv::waitKey(1);

              //  break ;
            }


    }




  std::cout<<"Pfps "<<fps_c/c<<std::endl;
 //engine.load(url);
    return app.exec();
}
