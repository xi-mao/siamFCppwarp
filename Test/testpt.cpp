#include "testpt.h"

const float testpt::CONTEXT_AMOUNT = 0.5f;
const float testpt::ANCHOR_RATIOS[testpt::ANCHOR_RATIOS_NUM] = { 0.33, 0.5, 1, 2, 3 };
const float testpt::ANCHOR_SCALES[testpt::ANCHOR_SCALES_NUM] = { 8 };

const std::string  testpt::windowing("cosine");
const float testpt::context_amount=0.5;
const float testpt::test_lr=0.52;
const float testpt::penalty_k=0.04;
const float testpt::window_influence=0.21;
testpt::testpt(QObject *parent):QObject(parent)
{

}
testpt::~testpt(){

}
// TODO: What is this?
int testpt::calculate_s_z() {
    float bb_half_perimeter = bounding_box.width + bounding_box.height;
    float w_z = bounding_box.width + CONTEXT_AMOUNT * bb_half_perimeter;
    float h_z = bounding_box.height + CONTEXT_AMOUNT * bb_half_perimeter;
    return round(sqrt(w_z * h_z));
}
torch::Tensor testpt::get_subwindow(cv::Mat frame, int exampler_size, int original_size) {
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
            padded_frame(cv::Rect(left_pad, 0, frame_size.width, top_pad)).setTo(channel_average);
        }
        if (bottom_pad) {
            padded_frame(cv::Rect(left_pad, top_pad + frame_size.height, frame_size.width, bottom_pad)).setTo(channel_average);
        }
        if (left_pad) {
            padded_frame(cv::Rect(0, 0, left_pad, padded_frame.rows)).setTo(channel_average);
        }
        if (right_pad) {
            padded_frame(cv::Rect(left_pad + frame_size.width, 0, right_pad, padded_frame.rows)).setTo(channel_average);
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



torch::Tensor testpt::convert_bbox(torch::Tensor loc) {
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
torch::Tensor testpt::xyxy2cxywh(torch::Tensor box) {
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


    std::cout<<box.sizes()<<std::endl;
     box=box[0];

      std::cout<<box.sizes()<<std::endl;
    torch::Tensor t0=  box.narrow(1,0,1);  //1维 第0 开始 取一排
    std::cout<<"t0 sizes"<<t0.sizes()<<std::endl;
    torch::Tensor t1=  box.narrow(1,1,1);
     std::cout<<"t1 sizes"<<t1.sizes()<<std::endl;
    torch::Tensor t2=  box.narrow(1,2,1);
    std::cout<<"t2 sizes"<<t2.sizes()<<std::endl;
    torch::Tensor t3= box.narrow(1,3,1);
    std::cout<<"t3 sizes"<<t3.sizes()<<std::endl;



torch::Tensor t=  torch::cat({ (t0 +t2) / 2,
                               (t1 + t3) / 2,
                               t2 -t0 + 1,
                              t3 - t1 + 1},-1);
std::cout<<"t sizes"<<t.sizes()<<std::endl;
return t;



}

std::vector<float>  testpt::xywh2cxywh(cv::Rect roi){  //box_wh
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
    target_pos.push_back( x0 +( x2 - 1) / 2);
    target_pos.push_back(  (x3 - 1) / 2);

    target_sz.push_back( x2);
    target_sz.push_back( x3);
    std::cout<<"xywh2cxywh"<<list<<std::endl;
    return list;


}
void testpt::testpt_load( QString path,torch::DeviceType dvic){


     std::string  s= path.toStdString();

  torch::jit::script::Module model= torch::jit::load( s, dvic);
  model.eval();
  //cv::Mat fram=cv::imread();
  //输入图像
    cv::Mat image = cv::imread("00001.jpg");
 cv::Rect roi=   cv::selectROI("nn",image);
 xywh2cxywh(roi);//# bbox in xywh format is given for initialization in case of tracking
 bounding_box = roi;
 channel_average = cv::mean(image);

 int Sz=calculate_s_z();
std::cout<<Sz<<std::endl;
/*
torch::Tensor z_crop= get_subwindow(image,z_size,Sz);

std::cout<<z_crop.sizes()<<std::endl;
 output = model.forward({z_crop}).toTensorList();
//auto outputs = model.forward({tensor_image}).toTuple();
std::cout<<output.size()<<std::endl;
 qDebug()<<"load_ok";
*/
float scale=0;
cv::Mat im_z_crop= get_crop(image,target_pos,target_sz,z_size,0,scale,channel_average,context_amount);
 //output = model.forward({im_z_crop}).toTensorList();
}


void testpt::testpt_loadtrack( QString path,torch::DeviceType dvic){


     std::string  s= path.toStdString();

  torch::jit::script::Module model= torch::jit::load( s, dvic);
  model.eval();

  //return;
  //cv::Mat fram=cv::imread();
  //输入图像
    auto image = cv::imread("00001.jpg");
    cv::imshow("666",image);
    cv::Size frame_size = image.size();

    // TODO: What are these?
    float s_z = calculate_s_z();
    float scale_z = z_size / s_z;
    int s_x = round(s_z * x_size / z_size);
    torch::Tensor x_crop = get_subwindow(image, x_size, s_x);
/*
    cv::Mat image_transfomed;
    cv::resize(image, image_transfomed, cv::Size(224, 224));
    cv::cvtColor(image, image_transfomed, cv::COLOR_BGR2RGB);


    // 图像转换为Tensor
    torch::Tensor tensor_image = torch::from_blob(image_transfomed.data, {image_transfomed.rows, image_transfomed.cols,3},torch::kByte);
    tensor_image = tensor_image.permute({2,0,1});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);

    tensor_image = tensor_image.unsqueeze(0);
    */
torch::Tensor f0=output[0];
torch::Tensor f1=output[1];

 std::cout<<model.modules().size()<<std::endl;
//auto outputs = module->forward(inputs).toTuple();
//torch::Tensor out1 = outputs->elements()[0].toTensor();
//torch::Tensor out2 = outputs->elements()[1].toTensor();
 std::cout<<output.size()<<std::endl;
 std::cout<<x_crop.sizes()<<std::endl;
std::cout<<f0.sizes()<<std::endl;
std::cout<<f1.sizes()<<std::endl;
 std::cout<<x_crop.device()<<std::endl;
std::cout<<f0.device()<<std::endl;

std::cout<<f1.device()<<std::endl;
std::vector<torch::IValue> output = model.forward({x_crop,f0,f1}).toTuple()->elements();
//auto outputs = model.forward({tensor_image}).toTuple();
std::cout<<output.size()<<std::endl;

  torch::Tensor bbox=output[1].toTensor();

    std::cout<<bbox.sizes()<<std::endl;
 qDebug()<<"load_ok";
 xyxy2cxywh(bbox);

}
cv::Mat testpt::get_crop(cv::Mat im, std::vector<float> target_pos, std::vector<float> target_sz,const int z_size,const int x_size,float &scale ,cv::Scalar avg_chans,float context_amount ){
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
cv::Mat testpt::get_subwindow_tracking(cv::Mat im,
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
      torch::Tensor  crop_xyxy = cxywh2xyxy(crop_cxywh);

// warpAffine transform matrix
float M_13 = crop_xyxy[0].item<float>();; //0维切片
float M_23 = crop_xyxy[1].item<float>();
float M_11 = ((crop_xyxy[2] - M_13) / (model_sz - 1)).item<float>();
float M_22 = ((crop_xyxy[3] - M_23) / (model_sz - 1)).item<float>();


std::vector<float> ten={M_11,0,M_13,0,M_22,
                                M_23};
torch::Tensor mat2x3 =torch::tensor(ten).reshape({2,3});
std::cout<<"mat2x3"<<mat2x3<<std::endl;
cv::Mat im_patch ;
cv::warpAffine(im,im_patch,
                        tensor2Mat(mat2x3) , cv::Size(model_sz, model_sz),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                         cv::BORDER_CONSTANT,
                        avg_chans);
cv::imshow("warpAffine",im_patch);
return im_patch;
}
torch::Tensor testpt::cxywh2xyxy( torch::Tensor box ){
/*
    box = np.array(box, dtype=np.float32)
        return np.concatenate([
            box[..., [0]] - (box[..., [2]] - 1) / 2, box[..., [1]] -
            (box[..., [3]] - 1) / 2, box[..., [0]] +
            (box[..., [2]] - 1) / 2, box[..., [1]] + (box[..., [3]] - 1) / 2
        ],
                              axis=-1)
            */

    std::cout<<"cxywh2xyxy box"<<box.sizes()<<std::endl;
    // box=box[1];
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


torch::Tensor tt=torch::tensor(t);
    std::cout<<"tt "<<tt<<std::endl;

    return tt;


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
cv::Mat testpt::tensor_to_imarray(torch::Tensor out_tensor,int img_h,int img_w){

    //s1:sequeeze去掉多余维度,(1,C,H,W)->(C,H,W)；s2:permute执行通道顺序调整,(C,H,W)->(H,W,C)
    out_tensor = out_tensor.squeeze().detach().permute({ 1, 2, 0 });
    out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8); //s3:*255，转uint8
    out_tensor = out_tensor.to(torch::kCPU); //迁移至CPU
    cv::Mat resultImg(img_h, img_w, CV_8UC3, out_tensor.data_ptr()); // 将Tensor数据拷贝至Mat
    return resultImg ;

}

cv::Mat testpt::tensor2Mat(torch::Tensor &i_tensor)
{
    int height = i_tensor.size(0), width = i_tensor.size(1);
    //i_tensor = i_tensor.to(torch::kF32);
    i_tensor = i_tensor.to(torch::kCPU);
    cv::Mat o_Mat(cv::Size(width, height), CV_32F, i_tensor.data_ptr());


    std::cout<<"o_Mat "<<o_Mat<<std::endl;
    return o_Mat;
}

