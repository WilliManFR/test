#include "samEncoder.h"
#include <iostream>

SamEncoder::SamEncoder(const std::string& encoder_filename)
    : ModelBase(encoder_filename) 
{
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Creating encoder model >>>>>");
    cudaSetDevice(0);
    cudaStreamCreate(&m_stream);
    LoadModel();
}

SamEncoder::~SamEncoder()
{
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Destroying encoder model >>>>>");
    if (d_input_ptr != nullptr)
        cudaFree(d_input_ptr);
    if (d_output_ptr != nullptr)
        cudaFree(d_output_ptr);
}

bool SamEncoder::mallocInputOutput()
{
    // std::cout << m_engine->getBindingName(0);
    nvinfer1::Dims input_dim = m_engine->getTensorShape("input");
    nvinfer1::Dims output_dim = m_engine->getTensorShape("output");

    // Create GPU buffers on device
    cudaMalloc((void**)&d_input_ptr, m_max_batchsize * // 1
        input_dim.d[1] *
        input_dim.d[2] *
        input_dim.d[3] * sizeof(float)); // TODO: sizeof(size_t) ??

    cudaMalloc((void**)&d_output_ptr, m_max_batchsize *
        output_dim.d[1] * 
        output_dim.d[2] * 
        output_dim.d[3] * sizeof(float)); // TODO: sizeof(size_t) ??

    m_context->setTensorAddress("input", d_input_ptr);
    m_context->setTensorAddress("output", d_output_ptr);

    return true;
}

const cv::Mat SamEncoder::transform(const cv::Mat &imageBGR)
{
    cv::Mat img;

    // resize to the input size of the newtork
    cv::resize(imageBGR, img, cv::Size(1024, 1024));

    // BGR image => RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3);

    // seems to normalize with special values for each channels
    const cv::Scalar m_mean = cv::Scalar(123.675, 116.28, 103.53);
	const cv::Scalar m_std = cv::Scalar(58.395, 57.12, 57.375);
    const cv::Mat mean = cv::Mat(img.size(), CV_32FC3, m_mean);
    img = img - mean;
    const cv::Mat std_mat = cv::Mat(img.size(), CV_32FC3, m_std);
    img = img / std_mat;
  
    return std::move(img);
}

void SamEncoder::preprocess(const cv::Mat &image)
{
    cv::Mat img = image.clone();
    img = transform(img);

    std::vector<cv::Mat> channels;
	cv::split(img, channels);

	size_t offset = 0;
	for(const auto &channel : channels)
	{   
		cudaMemcpy(d_input_ptr + offset, channel.data, 
					channel.total() * sizeof(float), cudaMemcpyHostToDevice);
		offset += channel.total();
	}
}

void SamEncoder::Forward()
{
    assert(m_engine != nullptr);

    m_context->enqueueV3(m_stream);

    cudaStreamSynchronize(m_stream);
}

const std::vector<float> SamEncoder::postProcess()
{
    std::vector<float> result(output_size);
    cudaMemcpyAsync(result.data(), d_output_ptr, m_batchsize *
        output_size * sizeof(float), cudaMemcpyDeviceToHost, m_stream);

    cudaStreamSynchronize(m_stream);

    return std::move(result);
}

const std::vector<float> SamEncoder::getFeature(const cv::Mat &img)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    const cv::Mat mattmp = img.clone();
    
    preprocess(mattmp);

    Forward();
  
    auto res = postProcess();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Image encoding: " << ms_time.count() << " ms" << std::endl;

    return std::move(res);
}
