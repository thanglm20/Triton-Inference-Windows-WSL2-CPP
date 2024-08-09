
#ifndef __IMAGE_UTILS_H__
#define __IMAGE_UTILS_H__


#include <string>
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

static std::vector<uint8_t> read_resize(const std::string& image_path, int w, int h) {
	auto img = cv::imread(image_path);
	cv::resize(img, img, cv::Size(w, h));
	cv::Mat flat = img.reshape(1, img.total() * img.channels());
	std::vector<uint8_t> inputVec = img.isContinuous() ? flat : flat.clone();
	return inputVec;
}

static std::vector<uint8_t> read_image(const std::string& image_path) {
	auto img = cv::imread(image_path);
	cv::Mat flat = img.reshape(1, img.total() * img.channels());
	std::vector<uint8_t> inputVec = img.isContinuous() ? flat : flat.clone();
	return inputVec;
}

static std::vector<float> convert_to_chw(const std::vector<uint8_t>& inputVec, int h, int w, int c) {
	std::vector<float> outputs(inputVec.size());

	std::vector<float> mean_{ 0.485, 0.456, 0.406};
	std::vector<float> std_{ 0.229, 0.224, 0.225};
	for (int row = 0; row < h; ++row) {
		for (int col = 0; col < w; ++col) {
			for (int i = 0; i < c; ++i) {
				int src_idx = h * row * c + col * c + i;
				int dst_idx = i * h * w + row * w + col;
				int value = inputVec.at(src_idx);
				outputs[dst_idx] = (value * 1.0 - mean_[i]) / std_[i];
			}		
		}
	}
	return outputs;
}

static std::vector<float> create_input(const std::string& img_path) {
	std::vector<uint8_t> input_uint8 = read_resize(img_path, 256, 256);
	std::vector<float> input_fp = convert_to_chw(input_uint8, 256, 256, 3);
}

static void mat2array(const cv::Mat& mat, std::vector<uint8_t>& output_data, const int dst_w = 0, const int dst_h = 0){
	cv::Mat img = mat.clone();
	if (dst_w != 0 && dst_h != 0) {
		cv::resize(img, img, cv::Size(dst_w, dst_h), cv::INTER_NEAREST);
	}
	cv::Mat flat = img.reshape(1, img.total() * img.channels());
	output_data = img.isContinuous() ? flat : flat.clone();
}

static void array2mat(uint8_t* buffer_rgb, const int src_w, const int src_h,
					cv::Mat& output_mat, const int dst_w = 0, const int dst_h = 0)
{
	output_mat = cv::Mat(src_h, src_w, CV_8UC3, buffer_rgb);
	if (dst_w != 0 && dst_h != 0) {
		cv::resize(output_mat, output_mat, cv::Size(dst_w, dst_h), cv::INTER_NEAREST);
	}
}

static int resizeBilinear(const uint8_t* img_src, uint16_t w_src, uint16_t h_src,
                    uint8_t* img_dest, uint16_t w_dest, uint16_t h_dest)
{
    if (w_dest == 0 || h_dest == 0)
    {
        printf("New width & height image must differ 0\n");
        return -1;
    }

    uint8_t depth = 3;

    double w_scale = w_src * 1.0 / w_dest;
    double h_scale = h_src * 1.0 / h_dest;

    uint16_t x_floor;
    uint16_t x_ceil;
    uint16_t y_floor;
    uint16_t y_ceil;

    uint8_t v1;
    uint8_t v2;
    uint8_t v3;
    uint8_t v4;

    uint8_t q1;
    uint8_t q2;
    uint8_t q[3] = { 0 };


    for (int i = 0; i < h_dest; i++)
        for (int j = 0; j < w_dest; j++)
        {
            double x = i * h_scale;
            double y = j * w_scale;

            x_floor = std::floor(x);
            x_ceil = std::min((uint16_t)(h_src - 1), (uint16_t)std::ceil(x));
            y_floor = std::floor(y);
            y_ceil = std::min((uint16_t)(w_src - 1), (uint16_t)std::ceil(y));

            if (x_ceil == x_floor && y_ceil == y_floor)
            {
                q[0] = img_src[(uint16_t)x * w_src * depth + (uint16_t)y * depth];
                q[1] = img_src[(uint16_t)x * w_src * depth + (uint16_t)y * depth + 1];
                q[2] = img_src[(uint16_t)x * w_src * depth + (uint16_t)y * depth + 2];
            }

            else if (x_ceil == x_floor)
            {
                for (uint8_t c = 0; c < depth; c++)
                {
                    q1 = img_src[(uint16_t)x * w_src * depth + y_floor * depth + c];
                    q2 = img_src[(uint16_t)x * w_src * depth + y_ceil * depth + c];
                    q[c] = q1 * (y_ceil - y) + q2 * (y - y_floor);
                }

            }
            else if (y_ceil == y_floor)
            {
                for (int c = 0; c < depth; c++)
                {
                    q1 = img_src[x_floor * w_src * depth + (uint16_t)y * depth + c];
                    q2 = img_src[x_ceil * w_src * depth + (uint16_t)y * depth + c];
                    q[c] = (q1 * (x_ceil - x)) + (q2 * (x - x_floor));
                }
            }
            else
            {
                for (uint8_t c = 0; c < depth; c++)
                {
                    v1 = img_src[x_floor * w_src * depth + y_floor * depth + c];
                    v2 = img_src[x_ceil * w_src * depth + y_floor * depth + c];
                    v3 = img_src[x_floor * w_src * depth + y_ceil * depth + c];
                    v4 = img_src[x_ceil * w_src * depth + y_ceil * depth + c];

                    q1 = v1 * (x_ceil - x) + v2 * (x - x_floor);
                    q2 = v3 * (x_ceil - x) + v4 * (x - x_floor);
                    q[c] = q1 * (y_ceil - y) + q2 * (y - y_floor);
                }
            }
            for (uint8_t c = 0; c < depth; c++)
            {
                img_dest[i * w_dest * depth + j * depth + c] = q[c];
            }
        }
    
    return 0;
}


static int* resizeBilinear_cpu(int32_t* pixels, int w, int h, int w2, int h2)
{
	int32_t* temp = new int32_t[w2*h2];
	int32_t a, b, c, d, x, y, index;
	float x_ratio = ((float)(w - 1)) / w2;
	float y_ratio = ((float)(h - 1)) / h2;
	float x_diff, y_diff, blue, red, green;
	int offset = 0;
	for (int i = 0; i<h2; i++)
	{
		for (int j = 0; j<w2; j++)
		{
			x = (int)(x_ratio * j);
			y = (int)(y_ratio * i);
			x_diff = (x_ratio * j) - x;
			y_diff = (y_ratio * i) - y;
			index = (y*w + x);
			a = pixels[index];
			b = pixels[index + 1];
			c = pixels[index + w];
			d = pixels[index + w + 1];

			// blue element
			// Yb = Ab(1-w)(1-h) + Bb(w)(1-h) + Cb(h)(1-w) + Db(wh)
			blue = (a & 0xff)*(1 - x_diff)*(1 - y_diff) + (b & 0xff)*(x_diff)*(1 - y_diff) +
				(c & 0xff)*(y_diff)*(1 - x_diff) + (d & 0xff)*(x_diff*y_diff);

			// green element
			// Yg = Ag(1-w)(1-h) + Bg(w)(1-h) + Cg(h)(1-w) + Dg(wh)
			green = ((a >> 8) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 8) & 0xff)*(x_diff)*(1 - y_diff) +
				((c >> 8) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 8) & 0xff)*(x_diff*y_diff);

			// red element
			// Yr = Ar(1-w)(1-h) + Br(w)(1-h) + Cr(h)(1-w) + Dr(wh)
			red = ((a >> 16) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 16) & 0xff)*(x_diff)*(1 - y_diff) +
				((c >> 16) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 16) & 0xff)*(x_diff*y_diff);

			temp[offset++] =
				0xff000000 |
				((((int32_t)red) << 16) & 0xff0000) |
				((((int32_t)green) << 8) & 0xff00) |
				((int32_t)blue);
		}
	}
	return temp;
}

#endif
