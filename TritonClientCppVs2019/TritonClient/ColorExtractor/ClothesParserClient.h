

#ifndef  __CLOTHES_PARSER_CLIENT_H__
#define __CLOTHES_PARSER_CLIENT_H__

#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include <unordered_map>
#include <numeric>
#include "triton_client_libs/grpc_client.h"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }


struct LayerInfo {
    std::string name{ "" };
    std::string type{""};
    LayerInfo(std::string&& name_, std::string&& type_)
        : name(name_), type(type_) {}
};

struct TritonClientConfigs {
    bool verbose = false;
    std::string url{ "172.31.123.244:8001" };
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
    bool use_cached_channel = true;
    
};

struct ModelConfigs {
    std::string model_name = "ensemble_segmentation";
    std::string model_version = "1";
    std::vector<LayerInfo> input_layers;
    std::vector<LayerInfo> output_layers;
};

class ClothesParserClient
{
public:
	ClothesParserClient(const TritonClientConfigs& client_configs);
	~ClothesParserClient();
	ClothesParserClient(const ClothesParserClient& other) = delete;
	ClothesParserClient(ClothesParserClient&& other) = delete;
	ClothesParserClient& operator =(const ClothesParserClient& other) = delete;
	ClothesParserClient& operator =(ClothesParserClient&& other) = delete;
    bool inferOne(uint8_t* input_buffer, const int in_len,
                  const int width, const int height,
                    std::vector<uint8_t>& output_buffer);
private:
    void initClient();

private:
    TritonClientConfigs m_client_configs;
    ModelConfigs m_model_configs;
    std::unique_ptr<tc::InferenceServerGrpcClient> m_client;
    tc::SslOptions m_ssl_options = tc::SslOptions();
};


#endif // ! __CLOTHES_PARSER_CLIENT_H__