
#include "YoloCrowdClient.h"

YoloCrowdClient::YoloCrowdClient(){
}

YoloCrowdClient::~YoloCrowdClient() {
}


bool YoloCrowdClient::initModelsParams() {
    
    m_model_configs = std::make_shared<ModelConfigs>();
    m_model_configs->model_name = "ensemble_yolov5_crowd";
    m_model_configs->model_version = "1";
    m_model_configs->batchsize = 1;
    auto input0 = std::make_shared<InputLayers>();
    input0->name = "INPUT";
    input0->data_type = "UINT8";
    int channel = 3;
    input0->len_data = m_model_configs->batchsize * m_iheight * m_iwidth * channel;
    input0->shape = std::vector<int64_t>{m_model_configs->batchsize, 
                                        static_cast<int64_t>(input0->len_data)};
    m_model_configs->input_layers.emplace_back(input0);
    auto output0 = std::make_shared<OutputLayers>();
    output0->name = "OUTPUT";
    output0->data_type = "FP32";
    output0->shape = std::vector<int64_t>{ 1, -1 , m_output_channel };
    m_model_configs->output_layers.emplace_back(output0);

    if (!mapModelLayers()) {
        std::cerr << "Mapping input, output layers failed";
        return false;
    }
    return true;
}


bool YoloCrowdClient::feedData(uint8_t** input_buffers, const std::vector<uint64_t>& data_lens){
    tc::Error err;
    // Reset the input for new request.
    err = m_model_configs->input_layers[0]->input_ptr->Reset();
    if (!err.IsOk()) {
        std::cerr << "Failed resetting input: " << err << std::endl;
        return false;
    }
    err = m_model_configs->input_layers[0]->input_ptr->AppendRaw(
                        input_buffers[0], data_lens[0] * sizeof(uint8_t));
    if (!err.IsOk()) {
        std::cerr << "Feeding data to Yolo Crowd model error: " << err << std::endl;
        return false;
    }
    return true;
}

bool YoloCrowdClient::getOutputModel(std::vector<float>& output_data){
    // Get pointers to the result returned...
    tc::Error err;
    float* output0_data;
    size_t out_size = 0;
    err = m_results_ptr->RawData(m_model_configs->output_layers[0]->name, 
                                (const uint8_t**)&output0_data, &out_size);
    if (!err.IsOk()) {
        std::cerr << "Getting output buffer 0 error: " << err << std::endl;;
        return false;
    }
    int size_data = out_size / sizeof(float);
    output_data.resize(size_data);
    std::copy(output0_data, output0_data + size_data, output_data.begin());
    return true;
}

bool YoloCrowdClient::inferOne(uint8_t* input_buffer, 
                    std::vector<float>& output_data) {
    std::vector<uint64_t> input_lens(m_model_configs->batchsize, 
                            m_model_configs->input_layers[0]->len_data);
    if (!feedData(&input_buffer, input_lens)) {
        std::cerr << "Feeding data to model failed!!!\n";
        return false;
    }
    if (!request()) {
        std::cerr << "Requesting to server failed!!!\n";
        return false;
    }
    getOutputModel(output_data);
    return true;
}