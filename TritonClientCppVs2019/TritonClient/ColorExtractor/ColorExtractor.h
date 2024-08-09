#ifndef  __COLOR_EXTRACTOR_H__
#define __COLOR_EXTRACTOR_H__

#define _CRT_SECURE_NO_DEPRECATE


#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include "common/ezxml.h"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include "ClothesParserClient.h"
#include "HumanParsingClient.h"

#define SHOW_FPS 1

typedef std::pair<std::vector<std::pair<std::string, float>>,
				std::vector<std::pair<std::string, float>>> OutputColors;

struct ColorInfo {
	int id;
	std::string name;
	int R, G, B;
	std::vector<std::vector<int>> lower;
	std::vector<std::vector<int>> upper;
	friend  std::ostream& operator<< (std::ostream& os, const ColorInfo& color) {
		os << "name: " << color.name
			<< ", RGB: " << color.R << ", " << color.G << ", " << color.B
			<< ", upper size: " << color.upper.size()
			<< ", lower size: " << color.lower.size()
			<< std::endl;
		return os;
	}
};

class ColorExtractor
{
public:
	ColorExtractor();
	//ColorExtractor(const TritonClientConfigs& client_configs, const std::string& colordef_file);
	~ColorExtractor();
	ColorExtractor(const ColorExtractor& other) = delete;
	ColorExtractor(ColorExtractor&& other) = delete;
	ColorExtractor& operator= (const ColorExtractor& other) = delete;
	ColorExtractor& operator= (ColorExtractor&& other) = delete;
	bool initialize(const TritonClientConfigs& client_configs,
					const std::string& colordef_file,
					const bool fake_inference = true);
	OutputColors getClothesColor(uint8_t* input_buffer_rgb,
						const int width, const int height);
	bool getManyClothesColors(std::vector<uint8_t*>& input_buffers_rgb, 
							const std::vector<std::pair<int, int>>& shapes,
							std::vector<OutputColors>& output_colors);
private:
	void runFakeInference();
	bool getColorDef(const std::string& colordef_file,
						std::set<int>& classList, 
						std::vector<ColorInfo>& colorInfoList);
	void extractColor(const cv::Mat& hsv_img, const cv::Mat& mask,
					std::vector<std::pair<std::string, float>>& upper_res,
					std::vector<std::pair<std::string, float>>& lower_res);
private:
	const uint16_t m_owidth = 256;
	const uint16_t m_oheight = 256;
	std::set<int> m_extractClass;
	std::vector<struct ColorInfo> m_colorInfo;
	//std::unique_ptr<ClothesParserClient> m_clothes_parser;
	std::unique_ptr<HumanParsingClient> m_human_parsing;

	std::vector<int> m_upper_seg_list{ 5, 6 ,7};
	std::vector<int> m_lower_seg_list{ 9, 10, 12 };
	int m_fake_inference{ 10 };

};

#endif
