#include "samDecoder.h"
#include <iostream>

SamDecoder::SamDecoder(const std::string& encoder_filename, const std::string& decoder_filename)
    : sam_encoder(encoder_filename), ModelBase(decoder_filename)
{
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Creating decoder model >>>>>");
    cudaSetDevice(0);
    cudaStreamCreate(&m_stream);
    LoadModel();
}

SamDecoder::~SamDecoder()
{
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Destroying decoder model >>>>>");

    if (d_input_image_embeddings_ptr != nullptr)
        cudaFree(d_input_image_embeddings_ptr);
    if (d_input_point_coords_ptr != nullptr)
        cudaFree(d_input_point_coords_ptr);
    if (d_input_point_labels_ptr != nullptr)
        cudaFree(d_input_point_labels_ptr);
    if (d_input_mask_input_ptr != nullptr)
        cudaFree(d_input_mask_input_ptr);
    if (d_input_has_mask_input_ptr != nullptr)
        cudaFree(d_input_has_mask_input_ptr);
    if (d_input_orig_im_size_ptr != nullptr)
        cudaFree(d_input_orig_im_size_ptr);
    
    
    if (d_output_low_res_masks_ptr != nullptr)
        cudaFree(d_output_low_res_masks_ptr);
    if (d_output_iou_predictions_ptr != nullptr)
        cudaFree(d_output_iou_predictions_ptr);
    if (d_output_masks_ptr != nullptr)
        cudaFree(d_output_masks_ptr);

    if (h_output_masks_ptr != nullptr)
        cudaFree(h_output_masks_ptr);
}

void SamDecoder::defineDynamicIOs(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config)
{
    // Définir des dimensions optimales
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{1, 1, 2});
    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3{1, 1, 2});
    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3{1, 10, 2});

    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 1});
    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{1, 1});
    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{1, 10});

    profile->setDimensions("orig_im_size", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {2}});
    profile->setDimensions("orig_im_size", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {2}});
    profile->setDimensions("orig_im_size", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {2}});
    config->addOptimizationProfile(profile);
}

bool SamDecoder::mallocInputOutput()
{
    // Create GPU buffers on device
    cudaMalloc((void**)&d_input_image_embeddings_ptr, m_max_batchsize * (256 * 64 * 64) * sizeof(float));
    cudaMalloc((void**)&d_input_point_coords_ptr, m_max_batchsize * (1 * 2) * sizeof(float));
    cudaMalloc((void**)&d_input_point_labels_ptr, m_max_batchsize * 1 * sizeof(float));
    cudaMalloc((void**)&d_input_mask_input_ptr, m_max_batchsize * (1 * 256 * 256) * sizeof(float));
    cudaMalloc((void**)&d_input_has_mask_input_ptr, m_max_batchsize * 1 * sizeof(float));
    cudaMalloc((void**)&d_input_has_mask_input_ptr, m_max_batchsize * 2 * sizeof(size_t));
    
    cudaMalloc((void**)&d_output_low_res_masks_ptr, m_max_batchsize *(4 * 256 * 256) * sizeof(float));
    cudaMalloc((void**)&d_output_iou_predictions_ptr, m_max_batchsize * (4) * sizeof(float));
    cudaMalloc((void**)&d_output_masks_ptr, m_max_batchsize * (4 * 1024 * 1024) * sizeof(float));

    h_output_masks_ptr = (float*)malloc(m_max_batchsize * (4 * 1024 * 1024) * sizeof(float));

    m_context->setTensorAddress("image_embeddings", d_input_image_embeddings_ptr);
    m_context->setTensorAddress("point_coords", d_input_point_coords_ptr);
    m_context->setTensorAddress("point_labels", d_input_point_labels_ptr);
    m_context->setTensorAddress("mask_input", d_input_mask_input_ptr);
    m_context->setTensorAddress("has_mask_input", d_input_has_mask_input_ptr);
    m_context->setTensorAddress("orig_im_size", d_input_orig_im_size_ptr);

    m_context->setTensorAddress("low_res_masks", d_output_low_res_masks_ptr);
    m_context->setTensorAddress("iou_predictions", d_output_iou_predictions_ptr);
    m_context->setTensorAddress("masks", d_output_masks_ptr);

    return true;
}

void SamDecoder::preprocess(const std::vector<float> &embedding, const float x, const float y, cv::Size orig_cv_im_size)
{

    cudaMemcpyAsync(d_input_image_embeddings_ptr, (float*)embedding.data(), 
					embedding.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    
    const float point_coord[] = {x, y};
    cudaMemcpyAsync(d_input_point_coords_ptr, point_coord, 
					2 * sizeof(float), cudaMemcpyHostToDevice, m_stream);

    const float point_labels[] = {1.0f};
    cudaMemcpyAsync(d_input_point_labels_ptr, point_labels, 
					1 * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    
    const std::vector<float> mask(256 * 256, 0.0f);
    cudaMemcpyAsync(d_input_mask_input_ptr, mask.data(), 
					256 * 256 * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    
    const float has_mask_input[] = {0.0f};
    cudaMemcpyAsync(d_input_has_mask_input_ptr, has_mask_input, 
					1 * sizeof(float), cudaMemcpyHostToDevice, m_stream);

    const size_t orig_im_size[] = {orig_cv_im_size.width, orig_cv_im_size.height};
    cudaMemcpyAsync(d_input_orig_im_size_ptr, orig_im_size, 
					2 * sizeof(size_t), cudaMemcpyHostToDevice, m_stream);
    
    cudaStreamSynchronize(m_stream);

}

void SamDecoder::Forward()
{
    assert(m_engine != nullptr);
    m_context->setInputShape("point_coords", nvinfer1::Dims3(1, 1, 2)); //point_coords: 1x1x2
    m_context->setInputShape("point_labels", nvinfer1::Dims2(1, 1));    //point_labels: 1x1

    m_context->enqueueV3(m_stream);

    cudaStreamSynchronize(m_stream);
}

const cv::Mat SamDecoder::postProcess()
{
    cudaMemcpyAsync(h_output_masks_ptr, d_output_masks_ptr, m_batchsize *
        (4 * 1024 * 1024) * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    int index = 0;
    auto outputMaskSam = cv::Mat(1024, 1024, CV_8UC1);
    for (int i = 0; i < outputMaskSam.rows; ++i) {
      for (int j = 0; j < outputMaskSam.cols; ++j) {
        auto val = h_output_masks_ptr[i * outputMaskSam.cols + j + index * 1024 * 1024] ;
        outputMaskSam.at<uchar>(i, j) = val > 0 ? 255 : 0;
      }
    }

    return std::move(outputMaskSam);
}

const cv::Mat SamDecoder::getMask(const cv::Mat &image, const float x, const float y)
{
 
    const std::vector<float> embedding = sam_encoder.getFeature(image);

    preprocess(embedding, x, y, image.size());

    Forward();

    auto res = postProcess();

    return std::move(res);
}

