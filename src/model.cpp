#include "model.h"

ModelBase::ModelBase(const std::string& model_path_basename)
    : m_onnx_file(model_path_basename + ".onnx"),
    m_engine_file(model_path_basename + ".engine")
{}

ModelBase::~ModelBase()
{
    cudaStreamSynchronize(m_stream);
    cudaStreamDestroy(m_stream);

    if (plan) delete plan;
}

bool ModelBase::buildEngine()
{
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Creating builder >>>>>");
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(m_logger);
    if (!builder)
        return false;

    const auto stronglyTypedFlag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    // const auto stronglyTypedFlag = 0;

    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Creating network >>>>>");
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(stronglyTypedFlag);
    if (!network)
        return false;

    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Creating parser >>>>>");
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, m_logger);
    if (!parser)
        return false;

    std::string parse_file_msg = "<<<<< Parsing file: " + m_onnx_file + " >>>>>";
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, parse_file_msg.c_str());
    parser->parseFromFile(m_onnx_file.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        m_logger.log(nvinfer1::ILogger::Severity::kERROR, parser->getError(i)->desc());
    }

    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Creating builder configuration >>>>>");
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    if (!config)
        return false;

    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Defining dynamic inputs/outputs >>>>>");
    defineDynamicIOs(builder, config);

    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Creating plan >>>>>");
    plan = builder->buildSerializedNetwork(*network, *config);

    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Saving plan >>>>>");
    SaveRtModel();

    delete parser;
    delete network;
    delete config;
    delete builder;

    return true;
}

void ModelBase::LoadModel()
{
    if (IsExists(m_engine_file) || IsExists(m_onnx_file) && !buildEngine())
        LoadTRTModel();
}

bool ModelBase::LoadTRTModel()
{
    if (!plan)
    {
        m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Loading plan >>>>>");
        std::ifstream fgie(m_engine_file, std::ios_base::in | std::ios_base::binary);
        if (!fgie)
            return false;

        m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Reading plan content >>>>>");
        std::stringstream buffer;
        buffer << fgie.rdbuf();

        m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Converting plan content to string >>>>>");
        std::string stream_model(buffer.str());

        deserializeCudaEngine(stream_model.data(), stream_model.size());
    }
    else
    {
        // if plan exists no need to read it from file
        deserializeCudaEngine(plan->data(), plan->size());
    }
    return true; 
}

bool ModelBase::deserializeCudaEngine(const void* blob, std::size_t size)
{
    // Need it to deserialize and run our inference
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Creating Runtime >>>>>");
    m_runtime = nvinfer1::createInferRuntime(m_logger);
    assert(m_runtime != nullptr);

    // Useless has we have no Plugins
    // Plugins are usefull if we want to adapt specific operations
    // that TensorRT doesn't natively support
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Initializing plugins >>>>>");
    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");

    // Deserialize Cuda Engine
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Deserializing Cuda Engine >>>>>");
    m_engine = m_runtime->deserializeCudaEngine(blob, size);
    assert(m_engine != nullptr);

    // Create an ExecutionContext to run an inference (Runtime) with
    // specific parameters.
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Creating execution context >>>>>");
    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    // Specify inputs/outputs for the run = context (as it's dynamic)
    m_logger.log(nvinfer1::ILogger::Severity::kINFO, "<<<<< Allocating memory for inputs/outputs >>>>>");
    mallocInputOutput();

    return true;
}