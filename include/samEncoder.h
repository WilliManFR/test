
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <stdexcept> 

#include "model.h"

#include <unordered_map>
#include "cuda_runtime.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
typedef unsigned char uint8;


class SamEncoder: ModelBase
{
public:
    SamEncoder(const std::string& encoder_filename);
    ~SamEncoder() override;
  
    const std::vector<float> getFeature(const cv::Mat &img);

private:
    SamEncoder(const SamEncoder &);

	bool mallocInputOutput() override;

    const cv::Mat transform(const cv::Mat &imageBGR);
	void preprocess(const cv::Mat &image);
	const std::vector<float> postProcess();
    void Forward() override;

	float* d_input_ptr;
	float* d_output_ptr;

	const size_t output_size = 256 * 64 * 64;

    SamEncoder &operator=(const SamEncoder &);
    static SamEncoder *instance;
};
