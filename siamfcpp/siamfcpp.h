#ifndef SIAMFCPP_H
#define SIAMFCPP_H

#include<opencv2/opencv.hpp>
//#undef slots
#include <torch/torch.h>
#include<torch/script.h>
//#define slots Q_SLOTS
//#include <QObject>
//#include<QDebug>
#include <cmath>

class siamfcpp //:public QObject
{
   // Q_OBJECT
public:
    siamfcpp( torch::DeviceType dvic=torch::DeviceType::CUDA);
    ~siamfcpp();
    //tracking model
     torch::jit::script::Module trackmodel;
 torch::jit::script::Module model;
    //ini value
     std::vector<float> target_pos, target_sz;

torch::List<torch::Tensor>  features;
cv::Mat  im_z_crop;
   cv::Scalar avg_chans;
   //device
    torch::DeviceType dvc;


   void ini( cv::Mat frame,cv::Rect roi);
   cv::Rect update( cv::Mat frame);

cv::Rect track( cv::Mat frame,std::vector<float> target_pos1,
            std::vector<float> target_sz1,
           torch::List<torch::Tensor> features,
            bool   update_state=true );
  std::vector<torch::Tensor> postprocess_score( torch::Tensor score,
                          torch::Tensor box_wh,
                          std::vector<float> target_sz1,
                          float scale_x
                          );
cv::Rect  restrict_box(cv::Rect tragrect);
torch::Tensor hann_window(int window_length,torch::DeviceType dev=torch::DeviceType::CPU);
  cv::Rect postprocess_box(int best_pscore_id,
                                   torch::Tensor score,
                                   torch::Tensor box_wh,
                                    std::vector<float> target_pos1,
                                    std::vector<float> target_sz1,
                                    float scale_x,
                                   float x_size,
                                    torch::Tensor penalty
                                   );
  torch::Tensor change(torch::Tensor r);


  torch::Tensor sz(torch::Tensor w,torch::Tensor h);

  torch::Tensor sz_wh(torch::Tensor wh);


int calculate_s_z();
torch::Tensor convert_bbox(torch::Tensor loc) ;

cv::Rect cxywh2xywh(cv::Rect roi) ;
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
std::vector<float> xywh2xyxy(cv::Rect roi );
 torch::Tensor cxywh2xyxy( torch::Tensor crop_cxywh );
// cv::Mat tensor_to_imarray(torch::Tensor out_tensor,int img_h,int img_w);
torch::Tensor Mat2tensor(cv::Mat im);
//torch::Tensor Matarr2tensor(cv::Mat im);
//torch::Tensor diagnoal(cv::Mat m);
cv::Mat tensor2Mat(torch::Tensor &i_tensor);
protected:


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





   cv::Rect bounding_box;


   // TODO: What are these?

   torch::List<torch::Tensor> zf;
   torch::Tensor anchors;
   torch::Tensor window;
   int  im_w;
  int im_h;
};

#endif // SIAMFCPP_H
