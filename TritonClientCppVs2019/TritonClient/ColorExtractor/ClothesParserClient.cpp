
#include "ClothesParserClient.h"


ClothesParserClient::ClothesParserClient(const TritonClientConfigs& client_configs) {
    m_client_configs = client_configs;
    initClient();
}

ClothesParserClient::~ClothesParserClient() {

}

void ClothesParserClient::initClient() {

    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    std::string err;
    if (m_client_configs.use_ssl) {
        m_client_configs.ssl_options.root_certificates = m_client_configs.root_certificates;
        m_client_configs.ssl_options.private_key = m_client_configs.private_key;
        m_client_configs.ssl_options.certificate_chain = m_client_configs.certificate_chain;
    }
    m_model_configs.input_layers.push_back({ "INPUT0", "UINT8" });
    m_model_configs.input_layers.push_back({ "INPUT1", "INT32" });
    m_model_configs.output_layers.push_back({ "OUTPUT0", "INT32" });
}

bool ClothesParserClient::inferOne(uint8_t* input_buffer, const int in_len,
                                    const int width, const int height,
                                    std::vector<uint8_t>& output_buffer) {

    int batchsize = 1;
    // Create GRPC connection
    tc::Error err;
    err = tc::InferenceServerGrpcClient::Create(
        &m_client, m_client_configs.url, m_client_configs.verbose, m_client_configs.use_ssl, 
        m_client_configs.ssl_options, tc::KeepAliveOptions(),
        m_client_configs.use_cached_channel);
    if (!err.IsOk()) {
        std::cerr << "Create gRPC client error!!!\n";
        return false;
    }
    // Mapping inputs
    std::vector<int64_t> shape0{ batchsize, in_len };
    tc::InferInput* input0;
    err = tc::InferInput::Create(&input0, m_model_configs.input_layers[0].name, shape0, m_model_configs.input_layers[0].type);
    if (!err.IsOk()) {
        std::cerr << "Creating input 0 error!!!\n";
        return false;
    }
    std::shared_ptr<tc::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    err = input0_ptr->AppendRaw(reinterpret_cast<uint8_t*>(input_buffer), in_len * sizeof(uint8_t));
    if (!err.IsOk()) {
        std::cerr << "Mapping input 0 error!!!\n";
        return false;
    }
    int input1_len = 3;
    std::vector<int64_t> shape1{ batchsize, input1_len };
    tc::InferInput* input1;
    err = tc::InferInput::Create(&input1, m_model_configs.input_layers[1].name, shape1, m_model_configs.input_layers[1].type);
    if (!err.IsOk()) {
        std::cerr << "Creating input 1 error!!!\n";
        return false;
    }
    std::shared_ptr<tc::InferInput> input1_ptr;
    input1_ptr.reset(input1);
    int channel = 3;
    std::vector<int32_t> in_shape{ height, width, channel };
    err = input1_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&in_shape[0]), in_shape.size() * sizeof(int32_t));
    if (!err.IsOk()) {
        std::cerr << "Mapping input 1 error!!!\n";
        return false;
    }
    // Generate the outputs to be requested.
    tc::InferRequestedOutput* output0;
    err = tc::InferRequestedOutput::Create(&output0, m_model_configs.output_layers[0].name);
    if (!err.IsOk()) {
        std::cerr << "Createing output 0 error!!!\n";
        return false;
    }
    std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
    output0_ptr.reset(output0);


    // The inference settings. Will be using default for now.
    tc::InferOptions options(m_model_configs.model_name);
    options.model_version_ = m_model_configs.model_version;
    options.client_timeout_ = m_client_configs.client_timeout;
    std::vector<tc::InferInput*> inputs = { input0_ptr.get(), input1_ptr.get() };
    std::vector<const tc::InferRequestedOutput*> outputs = { output0_ptr.get() };

    tc::InferResult* results;
    err = m_client->Infer(&results, options, inputs, outputs);
    if (!err.IsOk()) {
        std::cerr << "Client inference error!!!\n";
        return false;
    }
    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);
    // Get pointers to the result returned...
    int32_t* output0_data;
    size_t out_size = 0;
    err = results_ptr->RawData(m_model_configs.output_layers[0].name, (const uint8_t**)&output0_data, &out_size);
    if (!err.IsOk()) {
        std::cerr << "Getting output buffer 0 error!!!\n";
        return false;
    }
    int data_size = out_size / sizeof(int32_t);
    output_buffer.reserve(data_size);
    std::copy(output0_data, output0_data + data_size, output_buffer.begin());
    return true;
}

