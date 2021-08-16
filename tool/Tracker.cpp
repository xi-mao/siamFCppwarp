#include <cmath>
#include "Tracker.h"

const float Tracker::CONTEXT_AMOUNT = 0.5f;
const float Tracker::ANCHOR_RATIOS[Tracker::ANCHOR_RATIOS_NUM] = { 0.33, 0.5, 1, 2, 3 };
const float Tracker::ANCHOR_SCALES[Tracker::ANCHOR_SCALES_NUM] = { 8 };

// TODO: Move these util functions to their own place
// TODO: What is this?
torch::Tensor Tracker::change(torch::Tensor r) {
	return torch::max(r, 1 / r);
}

// TODO: What is this?
torch::Tensor Tracker::sz(torch::Tensor w, torch::Tensor h) {
	torch::Tensor pad = (w + h) / 2;
	return torch::sqrt((w + pad) * (h + pad));
}

// https://github.com/pytorch/pytorch/blob/1afc5841888ba324a533dde98ba94af637dac512/aten/src/ATen/native/TensorFactories.cpp#L881
// Why don't `mul_`, `cos_`, etc work?
torch::Tensor Tracker::hann_window(int window_length) {
	return torch::arange(window_length).mul(M_PI * 2. / static_cast<double>(window_length - 1)).cos().mul(-0.5).add(0.5).cuda();
}

std::vector<int> Tracker::unravel_index(int index, std::vector<int> shape) {
	std::vector<int> res;
	std::reverse(shape.begin(), shape.end());
	for (const int size : shape) {
		res.push_back(index % size);
		index /= size;
	}
	std::reverse(res.begin(), res.end());
	return res;
}

torch::List<torch::Tensor> Tracker::persist_only_last(torch::List<torch::Tensor> tensor_list) {
	return torch::List<torch::Tensor>{ tensor_list.get(tensor_list.size() - 1) };
}

// TODO: What is this?
int Tracker::calculate_s_z() {
	float bb_half_perimeter = bounding_box.width + bounding_box.height;
	float w_z = bounding_box.width + CONTEXT_AMOUNT * bb_half_perimeter;
	float h_z = bounding_box.height + CONTEXT_AMOUNT * bb_half_perimeter;
	return round(sqrt(w_z * h_z));
}

torch::Tensor Tracker::get_subwindow(cv::Mat frame, int exampler_size, int original_size) {
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
	).permute({ 0, 3, 1, 2 }).toType(torch::kFloat).cuda();
}

torch::Tensor Tracker::convert_score(torch::Tensor cls) {
	// TODO: What are these?
	torch::Tensor score = cls.permute({ 1, 2, 3, 0 }).contiguous().view({ 2, -1 }).permute({ 1, 0 });
	return score.softmax(1).narrow(1, 1, 1);
}

torch::Tensor Tracker::convert_bbox(torch::Tensor loc) {
	// TODO: What are these?
	torch::Tensor delta = loc.permute({ 1, 2, 3, 0 }).contiguous().view({ 4, -1 });
	delta.narrow(0, 0, 1) = delta.narrow(0, 0, 1) * anchors.narrow(1, 2, 1).t().cuda() + anchors.narrow(1, 0, 1).t().cuda();
	delta.narrow(0, 1, 1) = delta.narrow(0, 1, 1) * anchors.narrow(1, 3, 1).t().cuda() + anchors.narrow(1, 1, 1).t().cuda();
	delta.narrow(0, 2, 1) = torch::exp(delta.narrow(0, 2, 1)) * anchors.narrow(1, 2, 1).t().cuda();
	delta.narrow(0, 3, 1) = torch::exp(delta.narrow(0, 3, 1)) * anchors.narrow(1, 3, 1).t().cuda();
	return delta;
}

void Tracker::generate_anchors() {
	// TODO: What are these?
	int anchor_num = ANCHOR_RATIOS_NUM * ANCHOR_SCALES_NUM;
	int size = ANCHOR_STRIDE * ANCHOR_STRIDE;
	anchors = torch::zeros({ anchor_num, 4 });
	for (int i = 0; i < ANCHOR_RATIOS_NUM; i++) {
		int ws = sqrt(size / ANCHOR_RATIOS[i]);
		int hs = ws * ANCHOR_RATIOS[i];
		for (int j = 0; j < ANCHOR_SCALES_NUM; j++) {
			float s = ANCHOR_SCALES[j];
			float w = ws * s;
			float h = hs * s;
			anchors[i * ANCHOR_SCALES_NUM + j] = torch::tensor(torch::ArrayRef<float>({ -w / 2, -h / 2, w / 2, h / 2 }));
		}
	}
	torch::Tensor x1 = anchors.narrow(1, 0, 1);
	torch::Tensor y1 = anchors.narrow(1, 1, 1);
	torch::Tensor x2 = anchors.narrow(1, 2, 1);
	torch::Tensor y2 = anchors.narrow(1, 3, 1);
	anchors = torch::cat({ (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1 }, 1);
	anchors = anchors.repeat_interleave(SCORE_SIZE * SCORE_SIZE, 0);
	int ori = -(SCORE_SIZE / 2) * ANCHOR_STRIDE;
	torch::Tensor d = torch::tensor(torch::ArrayRef<int>({ ori }));
	for (int i = 1; i < SCORE_SIZE; i++) {
		d = torch::cat({ d, torch::tensor(torch::ArrayRef<int>({ ori + ANCHOR_STRIDE * i })) });
	}
	std::vector<torch::Tensor> meshgrid  = torch::meshgrid({ d, d });
	anchors.narrow(1, 0, 1) = meshgrid[1].flatten().repeat(anchor_num).reshape({ SCORE_SIZE * SCORE_SIZE * anchor_num, 1 });
	anchors.narrow(1, 1, 1) = meshgrid[0].flatten().repeat(anchor_num).reshape({ SCORE_SIZE * SCORE_SIZE * anchor_num, 1 });
	anchors.cuda();

	// window
	torch::Tensor hanning = hann_window(SCORE_SIZE);
	window = torch::ger(hanning, hanning).flatten().repeat(anchor_num).reshape({ SCORE_SIZE * SCORE_SIZE * anchor_num, 1 });
}

void Tracker::init(cv::Mat frame, cv::Rect roi, std::string obj_id, int obj_class_id, std::string obj_class_name, std::string aruco_marker_id) {
	bounding_box = roi;
	channel_average = cv::mean(frame);

	torch::Tensor z_crop = get_subwindow(frame, EXEMPLAR_SIZE, calculate_s_z());
    //torch::List<torch::Tensor> pre_zf = backbone_forward(z_crop);
    //zf = neck_forward(pre_zf);
modelforward(z_crop);
	ready_to_track = true;

	this->obj_id = obj_id;
	this->obj_class_id = obj_class_id;
	this->obj_class_name = obj_class_name;
	this->aruco_marker_id = aruco_marker_id;
}

torch::Tensor Tracker::get_penalty(float scale_z, torch::Tensor pred_bbox) {
	torch::Tensor s_c = change(
		sz(pred_bbox.narrow(0, 2, 1), pred_bbox.narrow(0, 3, 1)) / sz(torch::tensor(bounding_box.width * scale_z), torch::tensor(bounding_box.height * scale_z))
	);
	torch::Tensor r_c = change(
		torch::tensor((float)bounding_box.width / bounding_box.height) / (pred_bbox.narrow(0, 2, 1) / pred_bbox.narrow(0, 3, 1))
	);
	return torch::exp(-(r_c * s_c - 1) * TRACK_PENALTY_K).t();
}

int Tracker::get_best_idx(torch::Tensor penalty, torch::Tensor score) {
	torch::Tensor pscore = penalty * score * (1 - TRACK_WINDOW_INFLUENCE) + window * TRACK_WINDOW_INFLUENCE;
	return torch::argmax(pscore).item().toInt();
}

void Tracker::update_bbox(torch::Tensor pred_bbox, int best_idx, float scale_z, torch::Tensor penalty, torch::Tensor score, cv::Size frame_size) {
	torch::Tensor bbox = pred_bbox.narrow(1, best_idx, 1) / scale_z;
	float lr = (penalty[best_idx] * score[best_idx]).item().toFloat() * TRACK_LR;

	bounding_box.x = bbox[0].item().toFloat() + bounding_box.x + bounding_box.width / 2;
	bounding_box.y = bbox[1].item().toFloat() + bounding_box.y + bounding_box.height / 2;
	bounding_box.width = bounding_box.width * (1 - lr) + bbox[2].item().toFloat() * lr;
	bounding_box.height = bounding_box.height * (1 - lr) + bbox[3].item().toFloat() * lr;

	bounding_box.width = std::max(10, std::min(bounding_box.width, frame_size.width));
	bounding_box.height = std::max(10, std::min(bounding_box.height, frame_size.height));
	bounding_box.x = std::max(0, std::min(bounding_box.x, frame_size.width)) - bounding_box.width / 2;
	bounding_box.y = std::max(0, std::min(bounding_box.y, frame_size.height)) - bounding_box.height / 2;
}
