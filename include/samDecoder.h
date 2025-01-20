
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <stdexcept> 

#include "samEncoder.h"

#include <unordered_map>
#include "cuda_runtime.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
typedef unsigned char uint8;


class SamDecoder: ModelBase
{
public:
    SamDecoder(const std::string& encoder_filename, const std::string& decoder_filename);
    ~SamDecoder() override;
  
    const cv::Mat getMask(const cv::Mat &image, const float x, const float y);

private:
    SamDecoder(const SamDecoder &);

	void defineDynamicIOs(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config) override;
	bool mallocInputOutput() override;

	void preprocess(const std::vector<float> &embedding, float x, float y, cv::Size orig_cv_im_size);
	const cv::Mat postProcess();
    void Forward() override;

    SamEncoder sam_encoder; 

	float* d_input_image_embeddings_ptr;
	float* d_input_point_coords_ptr;
	float* d_input_point_labels_ptr;
	float* d_input_mask_input_ptr;
	float* d_input_has_mask_input_ptr;
	size_t* d_input_orig_im_size_ptr;

	float* d_output_low_res_masks_ptr;
	float* d_output_iou_predictions_ptr;
	float* d_output_masks_ptr;
	
	float* h_output_masks_ptr;

    SamDecoder &operator=(const SamDecoder &);
    static SamDecoder *instance;
};
