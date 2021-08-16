#include "siamfcpp.h"
siamfcpp::siamfcpp(std::string modelpth,torch::DeviceType dv)  {
    model=torch::jit::load(modelpth,dv);
std::cout<<"success";
    TRACK_PENALTY_K = 0.04;
    TRACK_WINDOW_INFLUENCE = 0.4;
    TRACK_LR = 0.5;
}

siamfcpp::~siamfcpp(){


}

track_result siamfcpp::track(cv::Mat frame){

    float s_z = calculate_s_z();
    float scale_z = EXEMPLAR_SIZE / s_z;
    int s_x = round(s_z * INSTANCE_SIZE / EXEMPLAR_SIZE);
    torch::Tensor x_crop = get_subwindow(frame, INSTANCE_SIZE, s_x);

//	torch::List<torch::Tensor> pre_xf = backbone_forward(x_crop);
    //torch::List<torch::Tensor> xf = neck_forward(pre_xf);

 std::vector<torch::IValue>   output =modelforward(x_crop);
    torch::Tensor loc = output[1].toTensor();
    torch::Tensor cls = output[2].toTensor();
   // torch::Tensor cls, loc;

   // cls /= (float)model.rpns.size();
    //loc /= (float)model.rpns.size();

    torch::Tensor score = convert_score(cls);
    torch::Tensor pred_bbox = convert_bbox(loc);
    torch::Tensor penalty = get_penalty(scale_z, pred_bbox);
    int best_idx = get_best_idx(penalty, score);
    update_bbox(pred_bbox, best_idx, scale_z, penalty, score, frame.size());

    track_result res;
    res.bbox = rectToRotatedRect(bounding_box);
    return res;
}
std::vector<torch::IValue>   siamfcpp::modelforward(torch::Tensor crop){
       std::vector<torch::IValue>   output = model.forward({crop}).toTuple()->elements();

std::cout<<output<<std::endl;
       return output;

}
