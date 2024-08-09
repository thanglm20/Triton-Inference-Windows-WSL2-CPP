
#include "pch.h"
#include  <algorithm>
#include <chrono>
#include "ColorExtractor.h"
#include "common/image_utils.h"
#include "common/ThreadPool.h"


ColorExtractor::ColorExtractor(){

}


ColorExtractor::~ColorExtractor(){}

bool ColorExtractor::initialize(const TritonClientConfigs& client_configs, 
								const std::string& colordef_file,
								const bool fake_inference) {
	m_human_parsing = std::make_unique<HumanParsingClient>();
	
	if (!m_human_parsing->initClient(client_configs)) {
		std::cerr << "Init Yolo client failed, check url!!!" << std::endl;;
		return false;
	}
	if (!m_human_parsing->initModelsParams()) {
		std::cerr << "Init Yolo model params failed, check configs!!!" << std::endl;;
		return false;
	}
	getColorDef(colordef_file, m_extractClass, m_colorInfo);
	for (const auto& color : m_colorInfo) {
		std::cout << color;
	}
	if (fake_inference) {
		runFakeInference();
	}

	return true;
}

bool ColorExtractor::getColorDef(const std::string& colordef_file, 
									std::set<int>& classList, 
									std::vector<ColorInfo>& colorInfoList){
	FILE* fp;
	//fp = fopen(colordef_file.c_str(), "rb");
	fopen_s(&fp, colordef_file.c_str(), "rb");
	ezxml_t colorXmlRoot = ezxml_parse_fp(fp);
	/*printf("[%s]\n", colorXmlRoot->name);
	printf("[%s]\n", colorXmlRoot->txt);*/

	ezxml_t colorXmlExtractClassList = ezxml_child(colorXmlRoot, "ExtractClassList");
	/*printf("[%s]\n", colorXmlExtractClassList->name);
	printf("%s\n", colorXmlExtractClassList->txt);*/

	std::string extractClass(colorXmlExtractClassList->txt);
	std::stringstream ss(extractClass);
	std::string temp;
	while (std::getline(ss, temp, ','))
	{
		classList.insert(std::stoi(temp));
	}
	ezxml_t colorXmlLists = ezxml_child(colorXmlRoot, "Lists");

	for (auto color = ezxml_child(colorXmlLists, "Color"); color; color = color->next)
	{
		struct ColorInfo colorInfo;
		colorInfo.id = atoi(ezxml_child(color, "ID")->txt);
		colorInfo.name.assign(ezxml_child(color, "Name")->txt);
		colorInfo.R = atoi(ezxml_child(color, "R")->txt);
		colorInfo.G = atoi(ezxml_child(color, "G")->txt);
		colorInfo.B = atoi(ezxml_child(color, "B")->txt);
		for (auto hsv = ezxml_child(color, "LowerHSV"); hsv; hsv = hsv->next) {
			colorInfo.lower.emplace_back(
							std::vector<int>{atoi(ezxml_child(hsv, "H")->txt),
										atoi(ezxml_child(hsv, "S")->txt),
										atoi(ezxml_child(hsv, "V")->txt)});
		}
		for (auto hsv = ezxml_child(color, "UpperHSV"); hsv; hsv = hsv->next) {
			colorInfo.upper.emplace_back(
				std::vector<int>{atoi(ezxml_child(hsv, "H")->txt),
								atoi(ezxml_child(hsv, "S")->txt),
								atoi(ezxml_child(hsv, "V")->txt)});
		}
		colorInfoList.push_back(colorInfo);
	}
	fclose(fp);
	return true;
}

static inline std::vector<std::string> getColorRange(int h, int s, int v, std::vector<ColorInfo>& colorInfoList) {
	std::vector<std::string> color_name;
	for (const auto& color : colorInfoList) {
		if (color.upper.size() != color.lower.size()) {
			return color_name;
		}
		for (int i = 0; i < color.upper.size(); ++i) {
			if ((color.lower[i][0] <= h && color.upper[i][0] >= h)
				&& (color.lower[i][1] <= s && color.upper[i][1] >= s)
				&& (color.lower[i][2] <= v && color.upper[i][2] >= v)) {
				color_name.push_back(color.name);
			}
		}
	}
	return color_name;
}

void ColorExtractor::extractColor(const cv::Mat& hsv_img, const cv::Mat& mask,
								std::vector<std::pair<std::string, float>>& upper_res,
								std::vector<std::pair<std::string, float>>& lower_res) {
	std::map<std::string, float> upper_color_map;
	std::map<std::string, float> lower_color_map;
	uint64_t upper_count{ 0 };
	uint64_t lower_count{ 0 };
	
	for (int r = 0; r < m_oheight; ++r) {
		for (int c = 0; c < m_owidth; ++c) {
			int m = mask.at<uchar>(r, c);
			auto value = hsv_img.at<cv::Vec3b>(r, c);
			int h = static_cast<int>(value[0]);
			int s = static_cast<int>(value[1]);
			int v = static_cast<int>(value[2]);
			//check upper clothes
			if (std::find(m_upper_seg_list.begin(), m_upper_seg_list.end(), m) != std::end(m_upper_seg_list)) {
				++upper_count;
				auto color_name = getColorRange(h, s, v, m_colorInfo);
				for (const auto& n : color_name) {
					upper_color_map[n]++;
				}
			}
			//check lower clothes
			else if(std::find(m_lower_seg_list.begin(), m_lower_seg_list.end(), m) != std::end(m_lower_seg_list)){
				++lower_count;
				auto color_name = getColorRange(h, s, v, m_colorInfo);
				for (const auto& n : color_name) {
					lower_color_map[n]++;
				}
			}
		}
	}

	float min_percent = 10.0;
	for (auto& upper : upper_color_map) {
		float percent = upper.second / upper_count * 100;
		if (percent > min_percent) {
			upper_res.push_back({ upper.first, percent });
		}
	}
	for (auto& lower : lower_color_map) {
		float percent = lower.second / upper_count * 100;
		if (percent > min_percent) {
			lower_res.push_back({ lower.first, percent });
		}
	}
	if (upper_res.size() == 0) {
		upper_res.push_back({ "unknown", 0.0 });
	}
	if (lower_res.size() == 0) {
		lower_res.push_back({ "unknown", 0.0 });
	}
}

OutputColors ColorExtractor::getClothesColor(uint8_t* input_buffer_rgb,
									const int width, const int height) {
	std::vector<std::pair<std::string, float>> upper_res;
	std::vector<std::pair<std::string, float>> lower_res;
	auto start1 = std::chrono::high_resolution_clock::now();
	std::vector<uint8_t> output;
	cv::Mat img;
	array2mat(input_buffer_rgb, width, height, img,
		m_human_parsing->getNetWidth(), m_human_parsing->getNetHeight());
	std::vector<uint8_t> buffer;
	mat2array(img, buffer);
	if (!m_human_parsing->inferOne(buffer.data(), output)) {
		std::cerr << "Triton client error!!!\n";
		return {};
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	auto dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
	auto start = std::chrono::high_resolution_clock::now();
	cv::Mat mask = cv::Mat(m_oheight, m_owidth, CV_8UC1, output.data());
	// convert input buffer RGB to cv::Mat
	cv::Mat hsv_img;
	cv::cvtColor(img, hsv_img, cv::COLOR_RGB2HSV);
	//Extract colors
	extractColor(hsv_img, mask, upper_res, lower_res);
	std::sort(upper_res.begin(), upper_res.end(),
		[](std::pair<std::string, float>& a, std::pair<std::string, float>& b) {return a.second > b.second; });
	std::sort(lower_res.begin(), lower_res.end(),
		[](std::pair<std::string, float>& a, std::pair<std::string, float>& b) {return a.second > b.second; });

	
	auto end = std::chrono::high_resolution_clock::now();
	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

#ifdef  SHOW_FPS
	std::cout << "Infer time: " << dur1.count() << "ms" << std::endl;
	std::cout << "Color time: " << dur.count() << "ms" << std::endl;
#endif //  SHOW_FPS
	return { upper_res, lower_res };
}


bool ColorExtractor::getManyClothesColors(std::vector<uint8_t*>& input_buffers_rgb,
	const std::vector<std::pair<int, int>>& shapes,
	std::vector <OutputColors>& output_colors) {
	if (input_buffers_rgb.size() != shapes.size()) {
		std::cerr << "Length of buffers and shape do not match!!!\n";
		return false;
	}
	int batch = input_buffers_rgb.size();
	ThreadPool pool(batch);
	std::vector<std::future<OutputColors> > results;
	for (int i = 0; i < batch; ++i) {
		int width = shapes[0].first;
		int height = shapes[0].first;
		auto buffer = input_buffers_rgb[i];
		results.emplace_back(pool.enqueue(
			[this, width, height, buffer]() {return getClothesColor(buffer, width, height); }));
	}
	//get outputs
	output_colors.reserve(batch);
	for (auto&& result : results)
		output_colors.push_back(result.get());	
	return true;
}


void ColorExtractor::runFakeInference() {
	int input_w = 256;
	int input_h = 256;
	std::vector<uint8_t> output;
	int size_input = input_w * input_h * 3;
	std::vector<uint8_t> input_buffer_rgb(size_input, 1);
	/*for (int i = 0; i < m_fake_inference; ++i) {
		if (!m_clothes_parser->inferOne(input_buffer_rgb.data(), size_input, input_w, input_h, output)) {
			std::cerr << "Triton client error!!!\n";
		}
	}*/

	for (int i = 0; i < m_fake_inference; ++i) {
		if (!m_human_parsing->inferOne(input_buffer_rgb.data(), output)) {
			std::cerr << "Triton client error!!!\n";
		}
	}
	std::cout << "Run fake inference succussfully!!!\n";
}