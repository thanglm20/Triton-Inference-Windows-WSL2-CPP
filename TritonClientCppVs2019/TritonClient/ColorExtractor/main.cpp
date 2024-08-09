#include "ColorExtractor.h"
#include "common/stb_image.h"
#include "common/args_parser.h"


using namespace parser;
using namespace std;


int main(int argc, char** argv) {

	Args::ADD_ARG_STRING("image_path", Desc("path to image file"), DefaultValue("D:/projects/TritonInferenceServer/deployment/images/1.jpg"));
	Args::ADD_ARG_STRING("colordef", Desc("path to ColorDef.xml file"), 
						DefaultValue("D:/projects/TritonInferenceServer/deployment/TritonClientCppVs2019/TritonClient/ColorDef.xml"));
	Args::ADD_ARG_STRING("url", Desc("url"), DefaultValue("172.31.123.244:8001"));
	Args::ADD_ARG_INT("num_test", Desc("number of loop testing"), DefaultValue("100"));

	Args::parseArgs(argc, argv);

	std::string image_path = Args::getStringValue("image_path");
	std::string colordef_file = Args::getStringValue("colordef");
	std::string url = Args::getStringValue("url");
	int numRuns = Args::getIntValue("num_test");

	TritonClientConfigs client_configs;
	client_configs.url = url;
	//auto extractor = std::make_unique<ColorExtractor>(client_configs, colordef_file);
	auto extractor = std::make_unique<ColorExtractor>();
	if (!extractor->initialize(client_configs, colordef_file)) {
		std::cout << "init color extractor error!!!";
		return -1;
	}

	// load image
	int w, h, comp, req_comp;
	uint8_t* img1 = stbi_load(image_path.c_str(), &w, &h, &comp, 3);
	std::cout << "w " << w << ", h " << h << std::endl;
	int img_size = w * h * 3;
	std::cout << "image size: " << img_size << std::endl;


	auto start = std::chrono::high_resolution_clock::now();
	std::vector<std::pair<std::string, float>> upper_res;
	std::vector<std::pair<std::string, float>> lower_res;
	for (int i = 0; i < numRuns; ++i) {
		upper_res.clear();
		lower_res.clear();
		OutputColors outputs = extractor->getClothesColor(img1, w, h);
		// Print results
		std::cout << "------------------------ \n";
		std::cout << "--- Upper: \n";

		for (auto& upper : outputs.first) {
			std::cout << upper.first << ": " << upper.second << "%" << std::endl;
		}
		std::cout << "--- Lower: \n";
		for (auto& lower : outputs.second) {
			std::cout << lower.first << ": " << lower.second << "%" << std::endl;
		}
	}
	// Test Batch images;
	auto start_batch = std::chrono::high_resolution_clock::now();
	int batch = 8;
	std::cout << "========== Testing batch input ================\n";
	std::vector < OutputColors> batch_outputs;
	std::vector<uint8_t*> batch_input(batch, img1);
	std::vector<std::pair<int, int>> shapes(batch, { w, h });
	extractor->getManyClothesColors(batch_input, shapes, batch_outputs);
	std::cout << "Output batch: " << batch_outputs.size() << std::endl;
	auto end_batch = std::chrono::high_resolution_clock::now();
	auto dur_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch);
	std::cout << "==== Time batch: " << dur_batch.count() << "ms \n";
	for (int i = 0; i < batch_outputs.size(); ++i) {
		std::cout << "------------------------ \n";
		std::cout << "--- Upper: \n";

		for (auto& upper : batch_outputs[i].first) {
			std::cout << upper.first << ": " << upper.second << "%" << std::endl;
		}
		std::cout << "--- Lower: \n";
		for (auto& lower : batch_outputs[i].second) {
			std::cout << lower.first << ": " << lower.second << "%" << std::endl;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "==== Average time on num_test (" << numRuns + batch << "): " << dur.count() / (numRuns + batch) << "ms" << std::endl;
	std::cout << "Done\n";
	return 0;
}

