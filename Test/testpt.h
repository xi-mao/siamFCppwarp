#ifndef TESTPT_H
#define TESTPT_H


#include<opencv2/opencv.hpp>
#undef slots
#include <torch/torch.h>
#include<torch/script.h>
#define slots Q_SLOTS
#include <QObject>
#include<QDebug>
class testpt:public QObject
{
    Q_OBJECT
public:
    testpt(QObject *parent=nullptr);
    ~testpt();
torch::List<torch::Tensor>  output;
   void testpt_load( QString path,torch::DeviceType dvic);
   void testpt_loadtrack( QString path,torch::DeviceType dvic);

   torch::Tensor get_subwindow(cv::Mat frame, int exampler_size, int original_size) ;
int calculate_s_z();
torch::Tensor convert_bbox(torch::Tensor loc) ;


torch::Tensor xyxy2cxywh(torch::Tensor box) ;
//void  postprocess_score();

std::list<double>  xywh2cxywh(cv::Rect roi); //box_wh


protected:
   static const float CONTEXT_AMOUNT;
   static const int EXEMPLAR_SIZE = 127;
   static const int INSTANCE_SIZE = 303;
   static const int ANCHOR_STRIDE = 8;
   static const int ANCHOR_RATIOS_NUM = 5;
   static const float ANCHOR_RATIOS[ANCHOR_RATIOS_NUM];
   static const int ANCHOR_SCALES_NUM = 1;
   static const float ANCHOR_SCALES[ANCHOR_SCALES_NUM];
   static const int TRACK_BASE_SIZE = 8;
   static const int SCORE_SIZE = (INSTANCE_SIZE - EXEMPLAR_SIZE) / ANCHOR_STRIDE + 1 + TRACK_BASE_SIZE;

   float TRACK_PENALTY_K;
   float TRACK_WINDOW_INFLUENCE;
   float TRACK_LR;

   cv::Rect bounding_box;
std::list<double> target_pos, target_sz;

   // TODO: What are these?
   cv::Scalar channel_average;
   torch::List<torch::Tensor> zf;
   torch::Tensor anchors;
   torch::Tensor window;
};

#endif // TESTPT_H
