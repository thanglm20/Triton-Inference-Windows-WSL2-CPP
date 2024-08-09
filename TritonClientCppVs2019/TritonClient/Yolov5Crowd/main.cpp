//#include <chrono>
//#include "common/args_parser.h"
//#include "opencv2/core.hpp"
//#include "opencv2/videoio.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#include "CrowdDetector.h"
//#include "common/image_utils.h"
//
//template <typename T>
//class simple_wrapper
//{
//public:
//	T value;
//};
//template <typename T>
//class fancy_wrapper
//{
//public:
//	fancy_wrapper(T const v) :value(v)
//	{
//	}
//	T const& get() const { return value; }
//	template <typename U>
//	U as() const
//	{
//		return static_cast<U>(value);
//	}
//private:
//	T value;
//};
//
//template <typename T, typename U,
//	template<typename> typename W = fancy_wrapper>
//class wrapping_pair
//{
//public:
//	wrapping_pair(T const a, U const b) :
//		item1(a), item2(b)
//	{
//	}
//	W<T> item1;
//	W<U> item2;
//};
//
//auto wa = wrapping_pair<int,int>(1, 2);
//using namespace parser;
//
//static std::unordered_map<int, cv::Scalar>
//COLOR_MAP{ {0, cv::Scalar(0, 255, 0)},
//		{1, cv::Scalar(0, 255, 255)} };
//
//void draw_boxes(cv::Mat& frame, const std::vector<Bbox> boxes) {
//	for (const auto& box : boxes) {
//		const auto& color = COLOR_MAP[box.id];
//		cv::rectangle(frame, cv::Point(box.xmin, box.ymin), 
//					cv::Point(box.xmax, box.ymax), color, 1);
//	}
//}
//
//int main(int argc, char** argv) {
//
//	Args::ADD_ARG_STRING("video_path", Desc("path to video file"), 
//			DefaultValue("D:/projects/TritonDeployment/data/1.mp4"));
//	Args::ADD_ARG_STRING("output_video", Desc("path to output file"),
//		DefaultValue("D:/projects/TritonDeployment/outputs/yolov5_crowd_output.avi"));
//	Args::ADD_ARG_STRING("url", Desc("url"), DefaultValue("172.31.123.244:8001"));
//
//	Args::parseArgs(argc, argv);
//
//	std::string video_path = Args::getStringValue("video_path");
//	std::string output_video = Args::getStringValue("output_video");
//	std::string url = Args::getStringValue("url");
//
//	cv::Mat frame;
//	cv::VideoCapture cap = cv::VideoCapture(video_path);
//	// check if we succeeded
//	if (!cap.isOpened()) {
//		cerr << "ERROR! Unable to open camera\n";
//		return -1;
//	}
//	int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
//	int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
//	int fps = cap.get(cv::CAP_PROP_FPS);
//	std::cout << "Video info: " << "width: " << frame_width << ", heigh: " 
//			<< frame_height << ", fps: " << fps << std::endl;
//	// Video Writer
//	cv::VideoWriter writer(output_video,
//		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));
//
//	// init Detector Client
//	auto detector = std::make_unique<CrowdDetector>();
//	TritonClientConfigs client_configs;
//	client_configs.url = url;
//	if (!detector->initialize(client_configs)) {
//		cerr << "ERROR! Unable to initialize Crowd \n";
//		return -1;
//	}
//
//	while (true) {
//
//		cap >> frame;
//		if (frame.empty())
//			break;
//		// TODO
//		std::vector<uint8_t> frame_buffer;
//		mat2array(frame, frame_buffer);
//		auto start = std::chrono::high_resolution_clock::now();
//		std::vector<Bbox> boxes;
//		detector->detect(frame_buffer.data(), frame_width, frame_height, boxes);
//		auto end = std::chrono::high_resolution_clock::now();
//		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//		std::cout << "Total spent time: " << dur.count() << std::endl;
//
//		int num_heads = std::count_if(boxes.begin(), boxes.end(),
//			[](const auto& b) -> bool {return (b.class_name == "head"); });
//		int num_people = std::count_if(boxes.begin(), boxes.end(),
//			[](const auto& b) -> bool {return (b.class_name == "person"); });
//		
//		std::cout << "Detection: " << boxes.size() 
//					<< ", num heads : " << num_heads
//					<< ", people : " << num_people << std::endl;
//		draw_boxes(frame, boxes);
//		cv::imshow("frame", frame);
//		writer.write(frame);
//		// Press  ESC on keyboard to exit
//		if ((char)27 == (char)cv::waitKey(1)) break;
//	}
//	writer.release();
//	cap.release();
//	std::cout << "Done\n";
//	return 0;
//}
//
