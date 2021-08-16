#ifndef SIAMFCPP_H
#define SIAMFCPP_H

#include <QObject>
#include"Tracker.h"


class siamfcpp : public Tracker
{

public:
   siamfcpp(std::string modelpth,torch::DeviceType dv) ;
~siamfcpp();


 std::vector<torch::IValue>  modelforward(torch::Tensor crop);
     virtual track_result track(cv::Mat frame);
private:
   TorchModule model;

signals:

};

#endif // SIAMFCPP_H
