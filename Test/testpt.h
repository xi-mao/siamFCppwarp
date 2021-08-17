#ifndef TESTPT_H
#define TESTPT_H


#include<opencv2/opencv.hpp>
#undef slots
#include <torch/torch.h>
#include<torch/script.h>
#define slots Q_SLOTS
#include <QObject>
#include<QDebug>
#include <cmath>
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

std::vector<float>  xywh2cxywh(cv::Rect roi); //box_wh
cv::Mat get_crop(cv::Mat im, std::vector<float> target_pos, std::vector<float> target_sz,
                 const int z_size,const int x_size,float &scale,
                 cv::Scalar avg_chans,float context_amount=0.5   );
cv::Mat get_subwindow_tracking(cv::Mat im,
                               std::vector<float> pos,
                             float  model_sz,
                             float  original_sz,
                              cv::Scalar avg_chans={0.0,0.0, 0.0, 0.0}
                              );
 torch::Tensor cxywh2xyxy( torch::Tensor crop_cxywh );
 cv::Mat tensor_to_imarray(torch::Tensor out_tensor,int img_h,int img_w);

cv::Mat tensor2Mat(torch::Tensor &i_tensor);
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


         static const int     total_stride=8;
         static const int   score_size=17;
        static const int  score_offset=87;
           static const float context_amount;
           static const float test_lr;
           static const float penalty_k;
           static const float window_influence;
      static const std::string    windowing;
       static const int   z_size=127;
      static const int    x_size=303;
        static const int  num_conv3x3=3;
      static const int   min_w=10;
       static const int   min_h=10;
     //    phase_init="feature",
      //   phase_track="track",
      //   corr_fea_output=False,



   float TRACK_PENALTY_K;
   float TRACK_WINDOW_INFLUENCE;
   float TRACK_LR;

   cv::Rect bounding_box;
std::vector<float> target_pos, target_sz;

   // TODO: What are these?
   cv::Scalar channel_average;
   torch::List<torch::Tensor> zf;
   torch::Tensor anchors;
   torch::Tensor window;
};

#endif // TESTPT_H
