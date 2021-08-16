#include "testpt.h"
#include <cmath>
const float testpt::CONTEXT_AMOUNT = 0.5f;
const float testpt::ANCHOR_RATIOS[testpt::ANCHOR_RATIOS_NUM] = { 0.33, 0.5, 1, 2, 3 };
const float testpt::ANCHOR_SCALES[testpt::ANCHOR_SCALES_NUM] = { 8 };

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


void testpt::testpt_load( QString path,torch::DeviceType dvic){


     std::string  s= path.toStdString();

  torch::jit::script::Module model= torch::jit::load( s, dvic);
  model.eval();
  //cv::Mat fram=cv::imread();
  //输入图像
    cv::Mat image = cv::imread("00001.jpg");
 cv::Rect roi=   cv::selectROI("nn",image);
 bounding_box = roi;
 channel_average = cv::mean(image);
 int Sz=calculate_s_z();
std::cout<<Sz<<std::endl;
torch::Tensor z_crop= get_subwindow(image,EXEMPLAR_SIZE,Sz);

std::cout<<z_crop.sizes()<<std::endl;
 output = model.forward({z_crop}).toTensorList();
//auto outputs = model.forward({tensor_image}).toTuple();
std::cout<<output.size()<<std::endl;
 qDebug()<<"load_ok";

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
    torch::Tensor t0=  box.narrow(1,0,1);  //0维 第0 开始 取一个
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

std::list<double>  testpt::xywh2cxywh(cv::Rect roi){  //box_wh
    /*
    rect = np.array(rect, dtype=np.float32)
        return np.concatenate([
            rect[..., [0]] + (rect[..., [2]] - 1) / 2, rect[..., [1]] +
            (rect[..., [3]] - 1) / 2, rect[..., [2]], rect[..., [3]]
        ],
                              axis=-1)
            */

    double x0=roi.x;
    double x1=roi.y;
    double x2=roi.width;
    double x3=roi.height;
    std::list<double> list={  x0 +( x2 - 1) / 2, x1 +
                           (x3 - 1) / 2, x2, x3
                       };
    target_pos.push_back( x0 +( x2 - 1) / 2);
    target_pos.push_back(  (x3 - 1) / 2);

    target_sz.push_back( x2);
    target_sz.push_back( x3);
    return list;


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
    float scale_z = EXEMPLAR_SIZE / s_z;
    int s_x = round(s_z * INSTANCE_SIZE / EXEMPLAR_SIZE);
    torch::Tensor x_crop = get_subwindow(image, INSTANCE_SIZE, s_x);
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
