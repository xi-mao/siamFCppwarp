#include "siamfcpp.h"
const float siamfcpp::CONTEXT_AMOUNT = 0.5f;
const float siamfcpp::ANCHOR_RATIOS[siamfcpp::ANCHOR_RATIOS_NUM] = { 0.33, 0.5, 1, 2, 3 };
const float siamfcpp::ANCHOR_SCALES[siamfcpp::ANCHOR_SCALES_NUM] = { 8 };

const std::string  siamfcpp::windowing("cosine");
const float siamfcpp::context_amount=0.5;
const float siamfcpp::test_lr=0.52;
const float siamfcpp::penalty_k=0.04;
const float siamfcpp::window_influence=0.21;
siamfcpp::siamfcpp( torch::DeviceType dvic)
{
    dvc=dvic;
    if(dvc==torch::DeviceType::CPU)
    {
      model= torch::jit::load( "../model/siamfcpp_features_cpu.pt");
      trackmodel= torch::jit::load( "../model/siamfcpp_track_cpu.pt");
    }
    else
    {
      model= torch::jit::load( "../model/siamfcpp_features_cuda.pt");
       trackmodel= torch::jit::load( "../model/siamfcpp_track_cuda.pt");
    }

      model.to(dvc);
      trackmodel.to(dvc);
      model.eval();
      trackmodel.eval();
}
siamfcpp::~siamfcpp(){

}
// TODO: What is this?
int siamfcpp::calculate_s_z() {
    float bb_half_perimeter = bounding_box.width + bounding_box.height;
    float w_z = bounding_box.width + CONTEXT_AMOUNT * bb_half_perimeter;
    float h_z = bounding_box.height + CONTEXT_AMOUNT * bb_half_perimeter;
    return round(sqrt(w_z * h_z));
}
torch::Tensor siamfcpp::get_subwindow(cv::Mat frame, int exampler_size, int original_size) {
    cv::Size frame_size = frame.size();

    // TODO: What are these?
    float s_z = original_size;
    float c = (s_z + 1) / 2;

    float context_xmin = floor((bounding_box.x + bounding_box.width / 2) - c + 0.5);
    float context_xmax = context_xmin + s_z - 1;
    float context_ymin = floor((bounding_box.y + bounding_box.height / 2) - c + 0.5);
    float context_ymax = context_ymin + s_z - 1;

    int left_pad = std::max(0.f, -context_xmin);
    int top_pad = std::max(0.f, -context_ymin);
    int right_pad = std::max(0.f, context_xmax - frame_size.width + 1);
    int bottom_pad = std::max(0.f, context_ymax - frame_size.height + 1);

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Mat frame_patch(context_ymax - context_ymin + 1, context_xmax - context_xmin + 1, frame.type());
    if (left_pad || top_pad || right_pad || bottom_pad) {
        cv::Mat padded_frame(frame.rows + top_pad + bottom_pad, frame.cols + left_pad + right_pad, frame.type());
        frame.copyTo(padded_frame(cv::Rect(left_pad, top_pad, frame.cols, frame.rows)));
        if (top_pad) {
            padded_frame(cv::Rect(left_pad, 0, frame_size.width, top_pad)).setTo(avg_chans);
        }
        if (bottom_pad) {
            padded_frame(cv::Rect(left_pad, top_pad + frame_size.height, frame_size.width, bottom_pad)).setTo(avg_chans);
        }
        if (left_pad) {
            padded_frame(cv::Rect(0, 0, left_pad, padded_frame.rows)).setTo(avg_chans);
        }
        if (right_pad) {
            padded_frame(cv::Rect(left_pad + frame_size.width, 0, right_pad, padded_frame.rows)).setTo(avg_chans);
        }
        padded_frame(cv::Rect(context_xmin, context_ymin, frame_patch.cols, frame_patch.rows)).copyTo(frame_patch);
    }
    else {
        frame(cv::Rect(context_xmin, context_ymin, frame_patch.cols, frame_patch.rows)).copyTo(frame_patch);
    }

    if (original_size != exampler_size) {
        cv::resize(frame_patch, frame_patch, cv::Size(exampler_size, exampler_size));
    }

    return torch::from_blob(
        frame_patch.data,
        { 1, frame_patch.rows, frame_patch.cols, 3 },
        torch::TensorOptions(torch::kByte)
    ).permute({ 0, 3, 1, 2 }).toType(torch::kFloat);
}



torch::Tensor siamfcpp::convert_bbox(torch::Tensor loc) {
    // TODO: What are these?
    torch::Tensor delta = loc.permute({ 1, 2, 3, 0 }).contiguous().view({ 4, -1 });
    delta.narrow(0, 0, 1) = delta.narrow(0, 0, 1) * anchors.narrow(1, 2, 1).t() + anchors.narrow(1, 0, 1).t();
    delta.narrow(0, 1, 1) = delta.narrow(0, 1, 1) * anchors.narrow(1, 3, 1).t() + anchors.narrow(1, 1, 1).t();
    delta.narrow(0, 2, 1) = torch::exp(delta.narrow(0, 2, 1)) * anchors.narrow(1, 2, 1).t();
    delta.narrow(0, 3, 1) = torch::exp(delta.narrow(0, 3, 1)) * anchors.narrow(1, 3, 1).t();
    return delta;
}
/*
def cxywh2xywh(box):
    box = np.array(box, dtype=np.float32)
    return np.concatenate([
        box[..., [0]] - (box[..., [2]] - 1) / 2, box[..., [1]] -
        (box[..., [3]] - 1) / 2, box[..., [2]], box[..., [3]]
    ],
                          axis=-1)
  */
torch::Tensor siamfcpp::xyxy2cxywh(torch::Tensor box) {
/*
    torch::cat([
        box( 0) - (box[..., [2]] - 1) / 2, box[..., [1]] -
        (box[..., [3]] - 1) / 2, box[..., [2]], box[..., [3]]
    ],

 C= np.concatenate([(bbox[..., [0]] + bbox[..., [2]]) / 2,
                    (bbox[..., [1]] + bbox[..., [3]]) / 2,
                    bbox[..., [2]] - bbox[..., [0]] + 1,
                    bbox[..., [3]] - bbox[..., [1]] + 1],
                   axis=-1)
                          axis=-1)
            */






    torch::Tensor t0=  box.narrow(1,0,1);  //1维 第0 开始 取一排

    torch::Tensor t1=  box.narrow(1,1,1);

    torch::Tensor t2=  box.narrow(1,2,1);

    torch::Tensor t3= box.narrow(1,3,1);




torch::Tensor t=  torch::cat({ (t0 +t2) / 2,
                               (t1 + t3) / 2,
                               t2 -t0 + 1,
                              t3 - t1 + 1},-1);

return t;



}
cv::Rect siamfcpp::cxywh2xywh(cv::Rect roi) {


    float x0=roi.x;
    float x1=roi.y;
    float x2=roi.width;
    float x3=roi.height;
 cv::Rect  list( x0 - (x2-1) / 2, x1 -
                            (x3 - 1) / 2, x2, x3

                       );

    return list;


}

std::vector<float>  siamfcpp::xywh2cxywh(cv::Rect roi){  //box_wh
    /*
    rect = np.array(rect, dtype=np.float32)
        return np.concatenate([
            rect[..., [0]] + (rect[..., [2]] - 1) / 2, rect[..., [1]] +
            (rect[..., [3]] - 1) / 2, rect[..., [2]], rect[..., [3]]
        ],
                              axis=-1)
            */

    float x0=roi.x;
    float x1=roi.y;
    float x2=roi.width;
    float x3=roi.height;
  std::vector<float> list={  x0 +( x2 - 1) / 2, x1 +
                           (x3 - 1) / 2, x2, x3
                       };
    target_pos.push_back( list[0]);
    target_pos.push_back( list[1]);

    target_sz.push_back( list[2]);
    target_sz.push_back( list[3]);
    //std::cout<<"xywh2cxywh"<<list<<std::endl;
    return list;


}
void siamfcpp::ini(cv::Mat frame,cv::Rect roi){ // return  features im_z_crop avg_chans


//
im_w=frame.cols;
im_h=frame.rows;

  std::cout<<"loadok"<<std::endl;

  xywh2cxywh(roi);//# bbox in xywh format is given for initialization in case of tracking
 bounding_box = roi;
 avg_chans = cv::mean(frame);

 /*  if self._hyper_params['windowing'] == 'cosine':
            window = np.outer(np.hanning(score_size), np.hanning(score_size))
            window = window.reshape(-1)
            */






torch::Tensor hannwindow = hann_window(score_size);


window=torch::outer(hannwindow,hannwindow).to(dvc);
window=window.reshape(-1);

 calculate_s_z();

/*
torch::Tensor z_crop= get_subwindow(image,z_size,Sz);

std::cout<<z_crop.sizes()<<std::endl;
 output = model.forward({z_crop}).toTensorList();
//auto outputs = model.forward({tensor_image}).toTuple();
std::cout<<output.size()<<std::endl;
 qDebug()<<"load_ok";
*/
  float scale=0;
 im_z_crop= get_crop(frame,target_pos,target_sz,z_size,0,scale,avg_chans,context_amount);

  torch::Tensor te=         Mat2tensor(im_z_crop).to(dvc);

features = model.forward({te}).toTensorList();
//std::cout<<"features"<<features.size()<<std::endl;




}
/*
torch::Tensor siamfcpp::diagnoal(cv::Mat m)
{
    std::vector<float> diagv;


        for(int i=0, j=0;i<m.rows;j++,i++)
        {
            float v=m.at<float>(i,j);
diagv.push_back(v);
if(i>=m.cols)
    break;



    }
        torch::Tensor t=torch::tensor(diagv).to(dvc);
      //  std::cout<<"t"<<t<<std::endl;
        return t;

}
*/
cv::Rect siamfcpp::update( cv::Mat frame){


/*
 rect = state  # bbox in xywh format is given for initialization in case of tracking
            box = xywh2cxywh(rect).reshape(4)
            target_pos_prior, target_sz_prior = box[:2], box[2:]
        features = self._state['features']
        */


  std::vector<float> target_pos_prior, target_sz_prior;
  target_pos_prior=target_pos;
  target_sz_prior=target_sz;

return track(frame,target_pos_prior,target_sz_prior,features);



}

cv::Rect siamfcpp::track( cv::Mat frame,std::vector<float> target_pos1,
            std::vector<float> target_sz1,
            torch::List<torch::Tensor>  features,
            bool   update_state ){
float scale_x=0;
cv::Mat im_x_crop= get_crop(frame,target_pos1,target_sz1,z_size,x_size,scale_x,avg_chans,context_amount).clone();
//std::cout<<"im_x_crop "<<im_x_crop.size()<<std::endl;
  torch::Tensor te=         Mat2tensor(im_x_crop);

//std::cout<<"featuressize "<<features.size()<<std::endl;
  torch::Tensor features1=features[0];
// std::cout<<"features1 "<<features1.sizes()<<std::endl;
   torch::Tensor features2=features[1];
// std::cout<<"features1 "<<features2.sizes()<<std::endl;
   std::vector<torch::IValue> output = trackmodel.forward({te.to(dvc),features1.to(dvc),features2.to(dvc)}).toTuple()->elements();

   //std::cout<<output.size()<<std::endl;






            torch::Tensor score=  output[0].toTensor().select(0,0).squeeze().to(dvc);
         //    std::cout<<"score sizes "<<score<<std::endl;
            torch::Tensor box=    output[1].toTensor().select(0,0).to(dvc);
            // std::cout<<"box sizes "<<box.sizes()<<std::endl;
             torch::Tensor cls=   output[2].toTensor().select(0,0).to(dvc);
            //  std::cout<<"cls sizes "<<cls.sizes()<<std::endl;
             torch::Tensor ctr=   output[3].toTensor().select(0,0).to(dvc);
          //  std::cout<<"ctr sizes "<<ctr.sizes()<<std::endl;
             torch::Tensor box_wh=   xyxy2cxywh(box.to(dvc));


   /*
    *   box = tensor_to_numpy(box[0])
        score = tensor_to_numpy(score[0])[:, 0]
        cls = tensor_to_numpy(cls[0])
        ctr = tensor_to_numpy(ctr[0])
# score post-processing
       best_pscore_id, pscore, penalty = self._postprocess_score(
           score, box_wh, target_sz, scale_x)
       # box post-processing
       new_target_pos, new_target_sz = self._postprocess_box(
           best_pscore_id, score, box_wh, target_pos, target_sz, scale_x,
           x_size, penalty)
           */
  torch::Tensor best_pscore_id, pscore, penalty ;
std::vector< torch::Tensor> rv=postprocess_score(
            score,box_wh,target_sz,scale_x
            );
best_pscore_id=rv[0].to(dvc);
int best_id=best_pscore_id.item<int>();
pscore=rv[1].to(dvc);
penalty=rv[2].to(dvc);
cv::Rect rc= postprocess_box(best_id,score,box_wh,target_pos1,target_sz1,scale_x,x_size,penalty);
//std::cout<<"bestid "<<best_id<<std::endl;
//std::cout<<"rc "<<rc<<std::endl;
rc=restrict_box(rc);
//update state
target_pos[0]=rc.x;
target_pos[1]=rc.y;
target_sz[0]=rc.width;
target_sz[1]=rc.height;
cv::Rect rr= cxywh2xywh(rc);
//std::vector<float> rr1=xywh2xyxy
return rr;

}


std::vector<torch::Tensor> siamfcpp::postprocess_score( torch::Tensor score,
                        torch::Tensor box_wh,
                        std::vector<float> target_sz1,
                        float scale_x
                            ){
       /*
# size penalty
      penalty_k = self._hyper_params['penalty_k']
      target_sz_in_crop = target_sz * scale_x
      s_c = change(
          sz(box_wh[:, 2], box_wh[:, 3]) /
          (sz_wh(target_sz_in_crop)))  # scale penalty
      r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                   (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
      penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
      pscore = penalty * score

      # ipdb.set_trace()
      # cos window (motion model)
      window_influence = self._hyper_params['window_influence']
      pscore = pscore * (
          1 - window_influence) + self._state['window'] * window_influence
      best_pscore_id = np.argmax(pscore)

      return best_pscore_id, pscore, penalty
              */
//(box_wh[:, 2], box_wh[:, 3]
    torch::Tensor w=  box_wh.select(1,2).to(dvc);
    torch::Tensor h=    box_wh.select(1,3).to(dvc);
 std::vector<float> target_sz_in_crop1 ={ target_sz1[0] * scale_x,target_sz1[1] * scale_x};
 torch::Tensor target_sz_in_crop= torch::tensor(target_sz_in_crop1).to(dvc);
 torch::Tensor s_c=change(sz(w,h)/sz_wh(target_sz_in_crop)).to(dvc);
 torch::Tensor r_c=change((target_sz_in_crop[0]/target_sz_in_crop[1])/
         (w/h)
         ).to(dvc);
 torch::Tensor penalty =torch::exp(-(r_c*s_c-1)*penalty_k).to(dvc);
  torch::Tensor pscore=penalty*score;
  pscore=pscore*(
              1-window_influence)+window*window_influence;
  torch::Tensor best_pscore_id =torch::argmax(pscore).to(dvc);
  std::vector<torch::Tensor> rv={best_pscore_id,pscore,penalty};
  return rv;


}
cv::Rect siamfcpp::postprocess_box(int best_pscore_id,
                                 torch::Tensor score,
                                 torch::Tensor box_wh,
                                  std::vector<float> target_pos1,
                                  std::vector<float> target_sz1,
                                  float scale_x,
                                 float x_size,
                                  torch::Tensor penalty
                                 ){
// std::cout<<"box_wh"<<box_wh.sizes()<<std::endl;
   // std::cout<<"bestid"<<best_pscore_id<<std::endl;
     torch::Tensor   pred_in_crop = box_wh.select(0,best_pscore_id) /scale_x;


       torch::Tensor  lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr;
       torch::Tensor  res_x = pred_in_crop[0] + target_pos1[0] - (int)(x_size / 2) / scale_x;
       torch::Tensor       res_y = pred_in_crop[1] + target_pos1[1] - (int)(x_size / 2) / scale_x;
       torch::Tensor    res_w = target_sz1[0] * (1 - lr) + pred_in_crop[2] * lr;
       torch::Tensor    res_h = target_sz1[1] * (1 - lr) + pred_in_crop[3] * lr;

       // np.array([res_x, res_y])
           //  np.array([res_w, res_h])

           cv::Rect newrect(res_x.item<float>(),res_y.item<float>(),res_w.item<float>(),res_h.item<float>());
           return newrect;

}
cv::Rect  siamfcpp::restrict_box(cv::Rect tragrect){

       tragrect.x = fmax(0, fmin(im_w, tragrect.x));
          tragrect.y = fmax(0, fmin(im_h,  tragrect.y));
          tragrect.width = fmax(min_w,
                            fmin(im_w,  tragrect.width ));
          tragrect.height= fmax(min_h,
                            fmin(im_h, tragrect.height));
          return tragrect;

}



cv::Mat siamfcpp::get_crop(cv::Mat im, std::vector<float> target_pos, std::vector<float> target_sz,const int z_size,const int x_size,float &scale ,cv::Scalar avg_chans,float context_amount ){
   /*
    Returns
      -------
          cropped & resized image, (output_size, output_size) if output_size provied,
          otherwise, (x_size, x_size, 3) if x_size provided, (z_size, z_size, 3) otherwise
      """
      wc = target_sz[0] + context_amount * sum(target_sz)
      hc = target_sz[1] + context_amount * sum(target_sz)
      s_crop = np.sqrt(wc * hc)
      scale = z_size / s_crop

      # im_pad = x_pad / scale
      if x_size is None:
          x_size = z_size
      s_crop = x_size / scale

      if output_size is None:
          output_size = x_size
      if mask is not None:
          im_crop, mask_crop = func_get_subwindow(im,
                                                  target_pos,
                                                  output_size,
                                                  round(s_crop),
                                                  avg_chans,
                                                  mask=mask)
          return im_crop, mask_crop, scale
      else:
          im_crop = func_get_subwindow(im, target_pos, output_size, round(s_crop),
                                       avg_chans)
          return im_crop, scale
                          */
    float wc= target_sz[0] + context_amount *(target_sz[0]+target_sz[1]);

    float hc= target_sz[1] + context_amount *(target_sz[0]+target_sz[1]);

    float   s_crop = sqrt(wc * hc);
  int output_size=0;
     scale=z_size / s_crop;
    if (x_size == 0)
    {
      output_size = z_size;
    s_crop = z_size / scale;
     }else
    {
     output_size=x_size;
       s_crop = x_size / scale;
    }

cv::Mat im_crop=get_subwindow_tracking(im,target_pos,output_size,round(s_crop),avg_chans);



return im_crop ;




}
cv::Mat siamfcpp::get_subwindow_tracking(cv::Mat im,
                               std::vector<float> pos,
                             float  model_sz,
                             float  original_sz,
                              cv::Scalar avg_chans
                               ){
    /*
    Returns
       -------
       numpy.array
           image patch within _original_sz_ in _im_ and  resized to _model_sz_, padded by _avg_chans_
           (model_sz, model_sz, 3)
       """
       crop_cxywh = np.concatenate(
           [np.array(pos), np.array((original_sz, original_sz))], axis=-1)
       crop_xyxy = cxywh2xyxy(crop_cxywh)
       # warpAffine transform matrix
       M_13 = crop_xyxy[0]
       M_23 = crop_xyxy[1]
       M_11 = (crop_xyxy[2] - M_13) / (model_sz - 1)
       M_22 = (crop_xyxy[3] - M_23) / (model_sz - 1)
       mat2x3 = np.array([
           M_11,
           0,
           M_13,
           0,
           M_22,
           M_23,
       ]).reshape(2, 3)
       im_patch = cv2.warpAffine(im,
                                 mat2x3, (model_sz, model_sz),
                                 flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=tuple(map(int, avg_chans)))
       if mask is not None:
           mask_patch = cv2.warpAffine(mask,
                                       mat2x3, (model_sz, model_sz),
                                       flags=(cv2.INTER_NEAREST
                                              | cv2.WARP_INVERSE_MAP))
           return im_patch, mask_patch
       return im_patch
                         */


    std::vector<float> xywh={pos[0],pos[1],original_sz,original_sz};
    torch::Tensor  crop_cxywh=torch::tensor(xywh);
      torch::Tensor  crop_xyxy = cxywh2xyxy(crop_cxywh.to(dvc));

// warpAffine transform matrix
float M_13 = crop_xyxy[0].item<float>();; //0维切片
float M_23 = crop_xyxy[1].item<float>();
float M_11 = ((crop_xyxy[2] - M_13) / (model_sz - 1)).item<float>();
float M_22 = ((crop_xyxy[3] - M_23) / (model_sz - 1)).item<float>();


std::vector<float> ten={M_11,0,M_13,0,M_22,
                                M_23};
torch::Tensor mat2x3 =torch::tensor(ten).reshape({2,3}).to(dvc);
//std::cout<<"mat2x3"<<mat2x3<<std::endl;
cv::Mat im_patch ;
cv::warpAffine(im,im_patch,
                        tensor2Mat(mat2x3) , cv::Size(model_sz, model_sz),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                         cv::BORDER_CONSTANT,
                        avg_chans);
cv::imshow("warpAffine",im_patch);
return im_patch;
}
torch::Tensor siamfcpp::cxywh2xyxy( torch::Tensor box ){
/*
    box = np.array(box, dtype=np.float32)
        return np.concatenate([
            box[..., [0]] - (box[..., [2]] - 1) / 2, box[..., [1]] -
            (box[..., [3]] - 1) / 2, box[..., [0]] +
            (box[..., [2]] - 1) / 2, box[..., [1]] + (box[..., [3]] - 1) / 2
        ],
                              axis=-1)
            */

  //  std::cout<<"cxywh2xyxy box"<<box.sizes()<<std::endl;

/*
      std::cout<<box.sizes()<<std::endl;
    torch::Tensor t0=  box.select(0,0);  //1维 第0 开始 取一排
  std::cout<<"t0 "<<(t0).item<float>()<<std::endl;
    torch::Tensor t1=  box.select(0,1);
  std::cout<<"t1 "<<(t1).item<float>()<<std::endl;
    torch::Tensor t2=  box.select(0,2);
  std::cout<<"t2 "<<(t2).item<float>()<<std::endl;
    torch::Tensor t3= box.select(0,3);
  std::cout<<"t3 "<<(t3).item<float>()<<std::endl;
torch::Tensor x0=t2 - 1;
  std::cout<<"x0 "<<(t2 - 1).item<float>()<<std::endl;

  */

   float t0=  box.select(0,0).item<float>();  //1维 第0 开始 取一排

   float t1=  box.select(0,1).item<float>();

   float t2=  box.select(0,2).item<float>();

   float t3= box.select(0,3).item<float>();



  std::vector<float> t={t0 - (t2 - 1) / 2,t1 -
                        (t3 - 1) / 2, t0 +
                        (t2 - 1) / 2, t1 + (t3 - 1) / 2
                    };


torch::Tensor tt=torch::tensor(t).to(dvc);
  //  std::cout<<"tt "<<tt<<std::endl;

    return tt;


}



std::vector<float> siamfcpp::xywh2xyxy(cv::Rect roi ){




    float x0=roi.x;
    float x1=roi.y;
    float x2=roi.width;
    float x3=roi.height;
  std::vector<float> list={
                            x0,x1,x2 + x0 - 1,
                            x3+ x1 - 1
                       };


    return list;


}


/*

def tensor_to_imarray(t):
    r"""
    Perform naive detach / cpu / numpy process and then transpose
    cast dtype to np.uint8
    :param t: torch.Tensor, (1, C, H, W)
    :return: numpy.array, (H, W, C)
    """
    arr = t.detach().cpu().numpy().astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.transpose(1, 2, 0)

*/
/*
cv::Mat siamfcpp::tensor_to_imarray(torch::Tensor out_tensor,int img_h,int img_w){

    //s1:sequeeze去掉多余维度,(1,C,H,W)->(C,H,W)；s2:permute执行通道顺序调整,(C,H,W)->(H,W,C)
    out_tensor = out_tensor.squeeze().detach().permute({ 1, 2, 0 });
    out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8); //s3:*255，转uint8
    out_tensor = out_tensor.to(torch::kCPU); //迁移至CPU
    cv::Mat resultImg(img_h, img_w, CV_8UC3, out_tensor.data_ptr()); // 将Tensor数据拷贝至Mat
    return resultImg ;

}
*/

cv::Mat siamfcpp::tensor2Mat(torch::Tensor &i_tensor)
{
    int height = i_tensor.size(0), width = i_tensor.size(1);
    //i_tensor = i_tensor.to(torch::kF32);
    i_tensor = i_tensor.to(torch::kCPU);
    cv::Mat o_Mat(cv::Size(width, height), CV_32F, i_tensor.data_ptr());


  //  std::cout<<"o_Mat "<<o_Mat<<std::endl;
    return o_Mat;
}

torch::Tensor siamfcpp::Mat2tensor(cv::Mat im){

  torch::Tensor ts=  torch::from_blob(
            im.data,
            { 1, im.rows, im.cols, im.channels() },
            torch::TensorOptions(torch::kByte)
        ).permute({ 0, 3, 1, 2 }).toType(torch::kFloat);

  return ts;
}


torch::Tensor siamfcpp::change(torch::Tensor r){

    return torch::maximum(r,1.0/r);

}

torch::Tensor siamfcpp::sz(torch::Tensor w,torch::Tensor h){
    torch::Tensor pad=(w+h)*0.5;
    torch::Tensor sz2=(w+pad)*(h+pad);
return torch::sqrt(sz2);
}

torch::Tensor siamfcpp::sz_wh(torch::Tensor wh){
    torch::Tensor pad=(wh[0]+wh[1])*0.5;
    torch::Tensor sz2=(wh[0]+pad)*(wh[1]+pad);
return torch::sqrt(sz2);
}
torch::Tensor siamfcpp::hann_window(int window_length ,torch::DeviceType dev) {
    return torch::arange(window_length).mul(M_PI * 2. / static_cast<double>(window_length - 1)).cos().mul(-0.5).add(0.5).to(dev);
}
