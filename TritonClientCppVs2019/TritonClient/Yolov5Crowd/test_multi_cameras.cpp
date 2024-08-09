//#include <chrono>
//#include "common/args_parser.h"
//#include "opencv2/core.hpp"
//#include "opencv2/videoio.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#include "CrowdDetector.h"
//#include "common/image_utils.h"
//#include "common/ThreadWorker.h"
//
//using namespace parser;
//
//class CameraThread : public Worker{
//
//public:
//	CameraThread(const std::string& url, const std::string& video_path)
//	: m_url(url), m_video_path(video_path){
//	}
//	CameraThread(const CameraThread& other){
//		m_video_path = other.m_video_path;
//		m_url = other.m_url;
//	}
//	void run() override {
//		int counter = 0;
//		cv::Mat frame;
//		cv::VideoCapture cap = cv::VideoCapture(m_video_path);
//		// check if we succeeded
//		if (!cap.isOpened()) {
//			cerr << "ERROR! Unable to open camera\n";
//			return;
//		}
//		int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
//		int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
//		int fps = cap.get(cv::CAP_PROP_FPS);
//		std::cout << "Video info: " << "width: " << frame_width << ", heigh: "
//			<< frame_height << ", fps: " << fps << std::endl;
//
//		// init Detector Client
//		auto detector = std::make_unique<CrowdDetector>();
//		TritonClientConfigs client_configs;
//		client_configs.url = m_url;
//		if (!detector->initialize(client_configs)) {
//			cerr << "ERROR! Unable to initialize Crowd \n";
//			return;
//		}
//		while (is_running()) {
//			cap >> frame;
//			if (frame.empty())
//				break;
//			// TODO
//			std::vector<uint8_t> frame_buffer;
//			mat2array(frame, frame_buffer);
//			auto start = std::chrono::high_resolution_clock::now();
//			std::vector<Bbox> boxes;
//			detector->detect(frame_buffer.data(), frame_width, frame_height, boxes);
//			auto end = std::chrono::high_resolution_clock::now();
//			auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//			std::cout << "Total spent time: " << dur.count() << std::endl;
//			//delay for expected fps
//			if (dur.count() < 1000.0 / m_fps) {
//				int delay = (1000.0 / m_fps) - dur.count();
//				std::this_thread::sleep_for(std::chrono::milliseconds(delay));
//				std::cout << "Delay " << delay << "ms" << "\n";
//			}
//
//			int num_heads = std::count_if(boxes.begin(), boxes.end(),
//				[](const auto& b) -> bool {return (b.class_name == "head"); });
//			int num_people = std::count_if(boxes.begin(), boxes.end(),
//				[](const auto& b) -> bool {return (b.class_name == "person"); });
//
//			std::thread::id this_id = std::this_thread::get_id();
//			stringstream ss;
//			ss << this_id;
//			string mystring = ss.str();
//			std::cout << "Thread ID: " << mystring << ": Detection: " << boxes.size()
//				<< ", num heads : " << num_heads
//				<< ", people : " << num_people << std::endl;
//		}
//		std::cout << "Stopped!!!" << std::endl;
//	}
//private:
//	void draw_boxes(cv::Mat& frame, const std::vector<Bbox> boxes) {
//		for (const auto& box : boxes) {
//			const auto& color = _color[box.id];
//			cv::rectangle(frame, cv::Point(box.xmin, box.ymin),
//				cv::Point(box.xmax, box.ymax), color, 1);
//		}
//	}
//private:
//	std::unordered_map<int, cv::Scalar>
//		_color{ {0, cv::Scalar(0, 255, 0)},
//				{1, cv::Scalar(0, 255, 255)} };
//	std::string m_video_path;
//	std::string m_url;
//	int m_fps = 25;
//};
//
//
//int main(int argc, char** argv) {
//
//	Args::ADD_ARG_STRING("video_path", Desc("path to video file"),
//		DefaultValue("D:/projects/TritonDeployment/data/1.mp4"));
//	Args::ADD_ARG_INT("num_cam", Desc("number of cameras"), DefaultValue("5"));
//	Args::ADD_ARG_STRING("url", Desc("url"), DefaultValue("172.31.123.244:8001"));
//
//	Args::parseArgs(argc, argv);
//
//	std::string video_path = Args::getStringValue("video_path");
//	std::string output_video = Args::getStringValue("output_video");
//	std::string url = Args::getStringValue("url");
//	int num_cam = Args::getIntValue("num_cam");
//	
//	std::vector< CameraThread> list_cameras;
//	for (int i = 0; i < num_cam; ++i) {
//		list_cameras.emplace_back(CameraThread(url, video_path));
//	}
//
//	for (int i = 0; i < num_cam; ++i) {
//		list_cameras[i].start();
//	}
//	while (1) {
//		std::this_thread::sleep_for(std::chrono::seconds(5));
//	}
//	return 0;
//}
//



#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

constexpr int ALIGNMENT = 64;

struct alignas(ALIGNMENT) AlignedData {
	AlignedData() { val = 0; }
	int val;
}; 

int main() {

	const int num_threads = std::thread::hardware_concurrency();
	srand((unsigned int)time(NULL));
	AlignedData aligned_data0{};
	AlignedData aligned_data1{};
	std::cout << "Address of aligned_data0 : " << &aligned_data0 << '\n';
	std::cout << "Address of aligned_data1: " << &aligned_data1 << '\n';
	std::cout << "Size block - " 
				<< (unsigned long long) & aligned_data1 					
				- (unsigned long long) & aligned_data0 << '\n';

	// create lambda function for computing
	auto worker = [](AlignedData& d) {
		const int count = 10000000;
		for (int i = 0; i < count; ++i) {
			d.val = (i + i) / 2 + rand();;
		}
	};

	// benchmark
	std::vector<std::thread> threads;
	auto start = std::chrono::high_resolution_clock::now();
	auto t0 = std::thread([&]() { worker(aligned_data0); });
	auto t1 = std::thread([&]() { worker(aligned_data1); });
	t0.join();
	t1.join();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "spent time: " << duration.count() << "ms \n";
}