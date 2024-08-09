//
//
//#include <iostream>
//#include <fstream>
//#include <memory>
//#include <string>
//#include <grpcpp/grpcpp.h>
//
//#include "../triton_client_libs/grpc_client.h"
//#include "common/image_utils.h"
//#include "common/stb_image.h"
//#include "common/stb_image_write.h"
//
//namespace tc = triton::client;
//
//#define FAIL_IF_ERR(X, MSG)                                        \
//  {                                                                \
//    tc::Error err = (X);                                           \
//    if (!err.IsOk()) {                                             \
//      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
//      exit(1);                                                     \
//    }                                                              \
//  }
//
//
//int main() {
//
//    bool verbose = false;
//    std::string url("172.31.123.244:8001");
//    tc::Headers http_headers;
//    uint32_t client_timeout = 0;
//    bool use_ssl = false;
//    std::string root_certificates;
//    std::string private_key;
//    std::string certificate_chain;
//    bool test_use_cached_channel = false;
//    bool use_cached_channel = true;
//
//    std::string model_name = "ensemble_segmentation";
//    std::string model_version = "1";
//       
//    std::unique_ptr<tc::InferenceServerGrpcClient> client;
//    tc::SslOptions ssl_options = tc::SslOptions();
//    std::string err;
//    if (use_ssl) {
//        ssl_options.root_certificates = root_certificates;
//        ssl_options.private_key = private_key;
//        ssl_options.certificate_chain = certificate_chain;
//        err = "unable to create secure grpc client";
//    }
//    else {
//        err = "unable to create grpc client";
//    }
//
//    int w, h, comp, req_comp;
//    uint8_t* img1 = stbi_load("D:/projects/TritonInferenceServer/deployment/images/1.jpg", &w, &h, &comp, 3);
//    std::cout << "w " << w << ", h " << h << std::endl;
//    int img_size = w * h * 3;
//    std::cout << "image size: " << img_size << std::endl;
//
//    // Run with the same name to ensure cached channel is not used
//    int numRuns = test_use_cached_channel ? 2 : 1;
//    numRuns = 100;
//    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < numRuns; ++i) {
//        FAIL_IF_ERR(
//            tc::InferenceServerGrpcClient::Create(
//                &client, url, verbose, use_ssl, ssl_options, tc::KeepAliveOptions(),
//                use_cached_channel),
//            err);
//        //std::vector<float> input0_data(256 * 256 * 3, 0.0);
//        //auto fp_vec = create_input("D:/projects/TritonInferenceServer/deployment/images/1.jpg");
//        
//
//        std::vector<int64_t> shape0{ 1, img_size };
//        tc::InferInput* input0;
//        FAIL_IF_ERR(
//            tc::InferInput::Create(&input0, "INPUT0", shape0, "UINT8"),
//            "unable to get INPUT0");
//        std::shared_ptr<tc::InferInput> input0_ptr;
//        input0_ptr.reset(input0);
//        FAIL_IF_ERR(
//            input0_ptr->AppendRaw(
//                reinterpret_cast<uint8_t*>(img1),
//                img_size * sizeof(uint8_t)),
//            "unable to set data for intput");
//
//        std::vector<int64_t> shape1{ 1, 3 };
//        tc::InferInput* input1;
//        FAIL_IF_ERR(
//            tc::InferInput::Create(&input1, "INPUT1", shape1, "INT32"),
//            "unable to get INPUT1");
//        std::shared_ptr<tc::InferInput> input1_ptr;
//        input1_ptr.reset(input1);
//        std::vector<int32_t> in_shape{ h, w, 3 };
//        FAIL_IF_ERR(
//            input1_ptr->AppendRaw(
//                reinterpret_cast<uint8_t*>(&in_shape[0]), 3 * sizeof(int32_t)),
//            "unable to set data for intput");
//
//
//        // Generate the outputs to be requested.
//        tc::InferRequestedOutput* output0;
//        FAIL_IF_ERR(
//            tc::InferRequestedOutput::Create(&output0, "OUTPUT0"),
//            "unable to get 'output'");
//        std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
//        output0_ptr.reset(output0);
//
//
//        // The inference settings. Will be using default for now.
//        tc::InferOptions options(model_name);
//        options.model_version_ = model_version;
//        options.client_timeout_ = client_timeout;
//
//
//        std::vector<tc::InferInput*> inputs = { input0_ptr.get(), input1_ptr.get() };
//        std::vector<const tc::InferRequestedOutput*> outputs = { output0_ptr.get()};
//    
//        tc::InferResult* results;
//        FAIL_IF_ERR(
//            client->Infer(
//                &results, options, inputs, outputs),
//            "unable to run model");
//        std::shared_ptr<tc::InferResult> results_ptr;
//        results_ptr.reset(results);
//
//
//        // Get pointers to the result returned...
//        uint8_t* output0_data;
//        size_t output0_byte_size;
//        FAIL_IF_ERR(
//            results_ptr->RawData(
//                "OUTPUT0", (const uint8_t**)&output0_data, &output0_byte_size),
//            "unable to get result data for 'OUTPUT0'");
//       
//        std::string output_dir = "D:/projects/TritonInferenceServer/deployment/TritonClientCppVs2019/TritonClient/x64/Release/";
//        stbi_write_png("D:/projects/TritonInferenceServer/deployment/TritonClientCppVs2019/TritonClient/x64/Release/mask.jpg", 256, 256, 1, output0_data, 0);
//        cv::Mat mask = cv::Mat(256, 256, CV_8UC1, output0_data);
//        cv::imwrite(output_dir + "mask_cv.jpg", mask);
//    }
//    auto end = std::chrono::high_resolution_clock::now();
//    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    std::cout << "Average time: " << dur.count() / numRuns <<  "ms" << std::endl;
//    std::cout << "Done\n";
//	return 0;
//}