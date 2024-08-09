#ifndef __CROWD_DETECTOR_H__
#define __CROWD_DETECTOR_H__

#include <unordered_map>
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "YoloCrowdClient.h"

const static std::unordered_map<int, std::string> LABELS{ {0, "head"}, { 1, "person" } };

struct Bbox {
	int id{ -1 };
	std::string class_name{ "" };
	int xmin{ -1 };
	int ymin{ -1 };
	int xmax{ -1 };
	int ymax{ -1 };
	float score{ 0.0 };
	explicit Bbox(int id_, const std::string& name_, 
		int xmin_, int ymin_, int xmax_, int ymax_, float score_)
	: id(id_), class_name(name_), 
	xmin(xmin_), ymin(ymin_), xmax(xmax_), ymax(ymax_), score(score_)
	{}
};

class CrowdDetector
{
public:
	CrowdDetector();
	~CrowdDetector();
	CrowdDetector(const CrowdDetector& other) = delete;
	CrowdDetector(CrowdDetector&& other) = delete;
	CrowdDetector& operator =(const CrowdDetector& other) = delete;
	CrowdDetector& operator =(CrowdDetector&& other) = delete;
	bool initialize(const TritonClientConfigs& client_configs);
	bool detect(uint8_t* input_buffer_rgb,
		const int width, const int height, std::vector<Bbox>& boxes);
private:
	std::unique_ptr<YoloCrowdClient> m_yolo_client;
};

#endif