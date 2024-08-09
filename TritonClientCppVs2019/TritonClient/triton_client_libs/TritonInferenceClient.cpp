#include "TritonInferenceClient.h"


TritonInferenceClient::TritonInferenceClient()
{
}

TritonInferenceClient::~TritonInferenceClient()
{
}

bool TritonInferenceClient::initClient(const TritonClientConfigs& client_configs) {
    m_client_configs = client_configs;
    if (m_client_configs.use_ssl) {
        m_client_configs.ssl_options.root_certificates = m_client_configs.root_certificates;
        m_client_configs.ssl_options.private_key = m_client_configs.private_key;
        m_client_configs.ssl_options.certificate_chain = m_client_configs.certificate_chain;
    }
    tc::Error err;
    std::cout << "Connecting to server: " << m_client_configs.url << std::endl;
    err = tc::InferenceServerGrpcClient::Create(
        &m_client, m_client_configs.url, m_client_configs.verbose,
        m_client_configs.use_ssl, m_client_configs.ssl_options,
        tc::KeepAliveOptions(), m_client_configs.use_cached_channel);
    if (!err.IsOk()) {
        std::cerr << "Create gRPC client error: " << err << std::endl;
        return false;
    }
    return true;
}

bool TritonInferenceClient::mapModelLayers() {
    tc::Error err;
    if (m_model_configs == nullptr) {
        std::cerr << "Error. Please init model configs!!!" << std::endl;
        return false;
    }
    // Mapping input layers
    m_inputs.reserve(m_model_configs->input_layers.size());
    for (auto& input_layer : m_model_configs->input_layers) {
        err = tc::InferInput::Create(&input_layer->input, input_layer->name, 
                                    input_layer->shape, input_layer->data_type);
        if (!err.IsOk()) {
            std::cerr << "Creating input error: " << err << std::endl;
            return false;
        }
        input_layer->input_ptr.reset(input_layer->input);
        m_inputs.emplace_back(input_layer->input_ptr.get());
    }
    // Mapping output layers
    m_outputs.reserve(m_model_configs->output_layers.size());
    for (auto& output_layer : m_model_configs->output_layers) {
        err = tc::InferRequestedOutput::Create(&output_layer->output, output_layer->name);
        if (!err.IsOk()) {
            std::cerr << "Createing output error: " << err << std::endl;
            return false;
        }
        output_layer->output_ptr.reset(output_layer->output);
        m_outputs.emplace_back(output_layer->output_ptr.get());
    }
    return true;
}

bool TritonInferenceClient::request() {
    tc::Error err;
    // The inference settings. Will be using default for now.
    tc::InferOptions options(m_model_configs->model_name);
    options.model_version_ = m_model_configs->model_version;
    options.client_timeout_ = m_client_configs.client_timeout;
    //options.request_id_ = std::to_string(m_send_count);
    tc::InferResult* result;
    err = m_client->Infer(&result, options, m_inputs, m_outputs);
    if (!err.IsOk()) {
        std::cerr << "Client inference error: " << err << std::endl;;
        return false;
    }
    m_results_ptr.reset(result);
    return true;
}