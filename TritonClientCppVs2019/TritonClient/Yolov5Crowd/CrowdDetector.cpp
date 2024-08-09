#include "CrowdDetector.h"
#include "common/image_utils.h"
#include <chrono>

CrowdDetector::CrowdDetector() {
}

CrowdDetector::~CrowdDetector() {
}

bool CrowdDetector::initialize(const TritonClientConfigs& client_configs) {
	m_yolo_client = std::make_unique<YoloCrowdClient>();
	if(!m_yolo_client->initClient(client_configs)){
		std::cerr << "Init Yolo client failed, check url!!!" << std::endl;;
		return false;
	}
	if(!m_yolo_client->initModelsParams()){
		std::cerr << "Init Yolo model params failed, check configs!!!" << std::endl;;
		return false;
	}
	return true;
}

static void getBoundingBoxes(const std::vector<float>& output_data,
						const int src_w, const int src_h,
						const int net_w, const int net_h,
						std::vector<Bbox>& boxes) {
	float x_scale = src_w * 1.0 / net_w;
	float y_scale = src_h * 1.0 / net_h;
	int output_channel = 6;
	for (int i = 0; i < output_data.size(); i = i + output_channel) {
		int xmin = std::lroundf(output_data.at(i) * x_scale);
		int ymin = std::lroundf(output_data.at(i + 1) * y_scale);
		int xmax = std::lroundf(output_data.at(i + 2) * x_scale);
		int ymax = std::lroundf(output_data.at(i + 3) * y_scale);
		float score = output_data.at(i + 4);
		int id = static_cast<int>(output_data.at(i + 5));
		std::string class_name = LABELS.at(id);
		boxes.push_back(Bbox(id, class_name, xmin, ymin, xmax, ymax, score));
	}
}

bool CrowdDetector::detect(uint8_t* input_buffer_rgb, 
				const int width, const int height, 
				std::vector<Bbox>& boxes) {
	boxes.clear();
	auto start = std::chrono::high_resolution_clock::now();
	// convert input buffer RGB to cv::Mat
	cv::Mat img = cv::Mat(height, width, CV_8UC3, input_buffer_rgb);
	cv::resize(img, img, cv::Size(m_yolo_client->getInputWidth(), m_yolo_client->getInputHeight()));
	std::vector<uint8_t> buffer;
	mat2array(img, buffer);
	auto end = std::chrono::high_resolution_clock::now();
	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "preprocessing time: " << dur.count() << std::endl;
	std::vector<float> output_data;
	m_yolo_client->inferOne(buffer.data(), output_data);
	getBoundingBoxes(output_data, width, height,
				m_yolo_client->getInputWidth(), m_yolo_client->getInputHeight(), boxes);
	return true;
}