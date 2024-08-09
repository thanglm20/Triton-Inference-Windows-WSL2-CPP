#ifndef  __HUMAN_PARSING_CLIENT_H__
#define __HUMAN_PARSING_CLIENT_H__

#include <numeric>
#include "triton_client_libs/TritonInferenceClient.h"

class HumanParsingClient : public TritonInferenceClient
{
public:
	HumanParsingClient();
	~HumanParsingClient();
	HumanParsingClient(const HumanParsingClient& other) = delete;
	HumanParsingClient(HumanParsingClient&& other) = delete;
	HumanParsingClient& operator =(const HumanParsingClient& other) = delete;
	HumanParsingClient& operator =(HumanParsingClient&& other) = delete;
	bool inferOne(uint8_t* input_buffer, std::vector<uint8_t>& output_data);

public:
	bool initModelsParams() override;
	bool feedData(uint8_t** input_buffers, const std::vector<uint64_t>& data_lens) override;
	bool getOutputModel(std::vector<float>& output_buffers) override { return false; };
	bool getOutputModel(std::vector<uint8_t>& output_buffers) override;
	int64_t getNetWidth() const { return m_width; }
	int64_t getNetHeight() const { return m_height; }

private:
	int64_t m_width = 256;
	int64_t m_height = 256;
};

#endif