#pragma once
#undef slots
#include <torch/script.h>

#define slots Q_SLOTS
#include <opencv2/opencv.hpp>

typedef torch::jit::script::Module TorchModule;

struct track_result {
    cv::RotatedRect bbox;
    cv::Mat mask;
    std::vector<cv::Mat> contours;
};

class Tracker {

    bool ready_to_track = false;
    std::string obj_id;
    int obj_class_id;
    std::string obj_class_name;
    std::string aruco_marker_id;
    virtual std::vector<torch::IValue>  modelforward(torch::Tensor input) = 0;
    //virtual torch::List<torch::Tensor> backbone_forward(torch::Tensor input) = 0;
   // virtual torch::List<torch::Tensor> neck_forward(torch::List<torch::Tensor> input) = 0;

    // TODO: What are these
    torch::Tensor change(torch::Tensor r);
    torch::Tensor sz(torch::Tensor w, torch::Tensor h);
    torch::Tensor hann_window(int window_length);

protected:
    // TODO: What are these?
    static const float CONTEXT_AMOUNT;
    static const int EXEMPLAR_SIZE = 127;
    static const int INSTANCE_SIZE = 255;
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

    Tracker() {
        generate_anchors();
    };

    cv::RotatedRect rectToRotatedRect(cv::Rect rect) {
        return cv::RotatedRect(
            cv::Point2f(rect.x + (float)rect.width / 2, rect.y + (float)rect.height / 2),
            rect.size(),
            0
        );
    }

    // TODO: What are these?
    cv::Scalar channel_average;
    torch::List<torch::Tensor> zf;
    torch::Tensor anchors;
    torch::Tensor window;

    torch::List<torch::Tensor> persist_only_last(torch::List<torch::Tensor> tensor_list);

    // TODO: What are these?
    void generate_anchors();
    int calculate_s_z();
    torch::Tensor get_subwindow(cv::Mat frame, int exampler_size, int original_size);
    torch::Tensor convert_score(torch::Tensor cls);
    torch::Tensor convert_bbox(torch::Tensor loc);
    torch::Tensor get_penalty(float scale_z, torch::Tensor pred_bbox);
    int get_best_idx(torch::Tensor penalty, torch::Tensor score);
    void update_bbox(torch::Tensor pred_bbox, int best_idx, float scale_z, torch::Tensor penalty, torch::Tensor score, cv::Size frame_size);
    std::vector<int> unravel_index(int index, std::vector<int> shape);

public:

    virtual void init(cv::Mat frame, cv::Rect roi, std::string obj_id = "", int obj_class_id = -1, std::string obj_class_name = "", std::string aruco_marker_id = "");
    // TODO: https://gitlab.kikaitech.io/kikai-ai/siam-trackers/issues/11
  //  virtual void load_networks_instantly() = 0;
    virtual track_result track(cv::Mat frame) = 0;

    bool is_ready_to_track() {
        return ready_to_track;
    }

    void stop_tracking() {
        // TODO: More proper cleanups
        ready_to_track = false;
    }

    std::string get_obj_id() {
        return obj_id;
    }

    int get_obj_class_id() {
        return obj_class_id;
    }

    std::string get_obj_class_name() {
        return obj_class_name;
    }

    cv::Rect get_bounding_box() {
        return bounding_box;
    }

    std::string get_aruco_marker_id() {
        return aruco_marker_id;
    }
};
