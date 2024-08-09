
#ifndef  __TRITON_INFERENCE_CLIENT_H__
#define __TRITON_INFERENCE_CLIENT_H__

#include "grpc_client.h"

namespace tc = triton::client;

struct InputLayers {
    std::string name{ "" };
    std::string data_type{ "UINT8" };
    std::vector<int64_t> shape{ 1, -1 };
    uint64_t len_data{ 0 };
	tc::InferInput* input = nullptr;
	std::shared_ptr<tc::InferInput> input_ptr = nullptr;
};

struct OutputLayers {
    std::string name{ "" };
    std::string data_type{ "UINT8" };
    std::vector<int64_t> shape{ 1, -1 };
	tc::InferRequestedOutput* output = nullptr;
    uint64_t len_data{ 0 };
	std::shared_ptr<tc::InferRequestedOutput> output_ptr = nullptr;
};

struct ModelConfigs {
    std::string model_name{ "" };
    std::string model_version{ "1" };
    int batchsize{ 1 };

    std::vector<std::shared_ptr<InputLayers>> input_layers;
    std::vector< std::shared_ptr<OutputLayers>> output_layers;
};

struct TritonClientConfigs {
    bool verbose = false;
    std::string url{""};
    tc::Headers http_headers;
    uint32_t client_timeout = 0;
    bool use_ssl = false;
    std::string root_certificates;
    std::string private_key;
    std::string certificate_chain;
    tc::SslOptions ssl_options;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;
    bool test_use_cached_channel = false;
    bool use_cached_channel = false;
};

class TritonInferenceClient
{
public:
	TritonInferenceClient();
	~TritonInferenceClient();
    TritonInferenceClient(const TritonInferenceClient& other) = delete;
    TritonInferenceClient(TritonInferenceClient&& other) = delete;
    TritonInferenceClient& operator =(const TritonInferenceClient& other) = delete;
    TritonInferenceClient& operator =(TritonInferenceClient&& other) = delete;

    virtual bool initClient(const TritonClientConfigs& client_configs);
    virtual bool initModelsParams() = 0;
    virtual bool feedData(uint8_t** input_buffers, const std::vector<uint64_t>& data_lens) = 0;
    virtual bool getOutputModel(std::vector<float>& output_buffers) = 0;
    virtual bool getOutputModel(std::vector<uint8_t>& output_buffers) = 0;

protected:
    bool mapModelLayers();
    bool request();

protected:
    TritonClientConfigs m_client_configs;
    std::shared_ptr<ModelConfigs> m_model_configs = nullptr;
	std::unique_ptr<tc::InferenceServerGrpcClient> m_client = nullptr;
	std::vector<tc::InferInput*> m_inputs;
	std::vector<const tc::InferRequestedOutput*> m_outputs;
    std::shared_ptr<tc::InferResult> m_results_ptr;

private:
    uint64_t m_send_count{ 0 };
};

#endif

