#include <assert.h>
#include <fstream>
#include <iostream>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"

#include "opencv2/opencv.hpp"

class ModelBase
{
public:
	ModelBase(const std::string& model_path_basename);
	virtual ~ModelBase();
private:
	class Logger : public nvinfer1::ILogger
	{
	public:
		void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
		{
			switch (severity)
			{
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
				std::cerr << "kINTERNAL_ERROR: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kERROR:
				std::cerr << "kERROR: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kWARNING:
				std::cerr << "kWARNING: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kINFO:
				std::cerr << "kINFO: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kVERBOSE:
				// std::cerr << "kVERBOSE: " << msg << std::endl;
				break;
			default:
				break;
			}
		}
	};

protected:
    bool IsExists(const std::string& file)
	{
		std::fstream f(file.c_str());
		return f.is_open();
	}

	void SaveRtModel()
	{
		std::ofstream outfile(m_engine_file, std::ios_base::out | std::ios_base::binary);
		outfile.write((const char*)plan->data(), plan->size());
		outfile.close();
	}

	virtual void defineDynamicIOs(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config) {};

	bool buildEngine();
    void LoadModel();
    bool LoadTRTModel();
    bool deserializeCudaEngine(const void* blob, std::size_t size);
    virtual bool mallocInputOutput() = 0;

	void preprocess(const cv::Mat &image);
	template <typename T> const T postProcess();
    virtual void Forward() = 0;

protected:
	const std::string m_onnx_file;
	const std::string m_engine_file;

	Logger m_logger;
	cudaStream_t m_stream;

    // Build phase
	nvinfer1::IHostMemory* plan { nullptr };

    // Runtime phase
	nvinfer1::IRuntime* m_runtime;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;

    int m_max_batchsize = 1;
    int m_batchsize = 1;
};