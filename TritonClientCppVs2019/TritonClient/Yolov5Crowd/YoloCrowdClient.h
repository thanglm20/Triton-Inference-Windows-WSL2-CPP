#ifndef  __YOLO_CROWD_CLIENT_H__
#define __YOLO_CROWD_CLIENT_H__

#include <numeric>
#include "triton_client_libs/TritonInferenceClient.h"

class YoloCrowdClient : public TritonInferenceClient
{
public:
	YoloCrowdClient();
	~YoloCrowdClient();
	YoloCrowdClient(const YoloCrowdClient& other) = delete;
	YoloCrowdClient(YoloCrowdClient&& other) = delete;
	YoloCrowdClient& operator =(const YoloCrowdClient& other) = delete;
	YoloCrowdClient& operator =(YoloCrowdClient&& other) = delete;
	bool inferOne(uint8_t* input_buffer, std::vector<float>& output_data);

public:
	bool initModelsParams() override;
	bool feedData(uint8_t** input_buffers, const std::vector<uint64_t>& data_lens) override;
	bool getOutputModel(std::vector<float>& output_buffers) override;
	bool getOutputModel(std::vector<uint8_t>& output_buffers) override { return false; };

	int64_t getInputHeight() const {
		return m_iheight;
	}
	int64_t getInputWidth() const {
		return m_iwidth;
	}
	int64_t getOutputChannel() const {
		return m_output_channel;
	}
private: 
	int64_t m_iwidth = 640;
	int64_t m_iheight = 384;
	int64_t m_output_channel = 6;
};

#endif