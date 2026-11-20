// Standalone TensorRT 10 / CUDA replay for FP32 NPY network tensors; no CAD.
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

void check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void cudaCheck(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }
struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << m << '\n';
  }
};
struct Array { nvinfer1::Dims shape{}; std::vector<float> values; };
Array readNpy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  char magic[8]; f.read(magic, 8);
  check(f && std::string(magic, 8) == std::string("\x93NUMPY\x01\x00", 8), "Require NPY v1: " + path);
  std::uint16_t size; f.read(reinterpret_cast<char*>(&size), 2);
  std::string header(size, ' '); f.read(header.data(), size);
  check(header.find("'<f4'") != std::string::npos && header.find("'fortran_order': False") != std::string::npos,
        "Require little-endian FP32 C-order NPY");
  auto start = header.find('(', header.find("'shape'")); auto end = header.find(')', start);
  check(start != std::string::npos && end != std::string::npos, "Missing NPY shape");
  auto dims = header.substr(start+1, end-start-1); std::regex number("[0-9]+");
  Array a; std::size_t count = 1;
  for (std::sregex_iterator i(dims.begin(), dims.end(), number), last; i != last; ++i) {
    check(a.shape.nbDims < 8, "Too many dimensions");
    auto n = std::stoll(i->str()); check(n > 0 && n <= 2147483647, "Invalid dimension");
    check(count <= (std::size_t(1) << 32) / n, "NPY allocation too large");
    a.shape.d[a.shape.nbDims++] = n; count *= n;
  }
  check(a.shape.nbDims > 0, "Scalar inputs unsupported");
  a.values.resize(count); f.read(reinterpret_cast<char*>(a.values.data()), count*sizeof(float));
  check(bool(f) && f.peek() == std::char_traits<char>::eof(), "NPY data size mismatch"); return a;
}
void writeNpy(const std::filesystem::path& path, const Array& a) {
  std::ostringstream dims;
  for (int i=0;i<a.shape.nbDims;++i) dims << a.shape.d[i] << ", ";
  std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims.str() + "), }";
  h.append((64-((10+h.size()+1)%64))%64, ' '); h += '\n';
  std::ofstream f(path, std::ios::binary); const auto n=static_cast<std::uint16_t>(h.size());
  f.write("\x93NUMPY\x01\x00",8); f.write(reinterpret_cast<const char*>(&n),2); f.write(h.data(),h.size());
  f.write(reinterpret_cast<const char*>(a.values.data()),a.values.size()*sizeof(float)); f.close();
  check(bool(f), "Cannot write " + path.string());
}
struct Buffer {
  void* p=nullptr;
  ~Buffer() { if (p) cudaFree(p); }
};
struct Stream {
  cudaStream_t s{};
  Stream() { cudaCheck(cudaStreamCreate(&s)); }
  ~Stream() { cudaStreamDestroy(s); }
};
int main(int argc,char** argv) {
  try {
    check(argc==5,"Usage: replay_network ENGINE INPUT_A.npy INPUT_B.npy NEW_OUTPUT_DIR");
    std::ifstream file(argv[1],std::ios::binary);
    check(bool(file),"Cannot read engine");
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
    Logger logger; std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    check(bool(runtime),"Cannot create runtime");
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(),bytes.size()));
    check(bool(engine),"Cannot deserialize engine");
    std::unique_ptr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext());
    check(bool(ctx),"Cannot create context");
    std::map<std::string,Array> arrays;
    arrays["inputA"]=readNpy(argv[2]); arrays["inputB"]=readNpy(argv[3]);
    check(ctx->setInputShape("inputA",arrays.at("inputA").shape),"inputA shape rejected");
    check(ctx->setInputShape("inputB",arrays.at("inputB").shape),"inputB shape rejected");
    std::map<std::string,Buffer> buffers; Stream stream;
    for (int i=0;i<engine->getNbIOTensors();++i) {
      std::string name=engine->getIOTensorName(i);
      check(name.find('/')==std::string::npos && name.find("..") == std::string::npos,"Unsafe tensor name");
      check(engine->getTensorDataType(name.c_str())==nvinfer1::DataType::kFLOAT,"Require FP32 IO");
      check(engine->getTensorLocation(name.c_str())==nvinfer1::TensorLocation::kDEVICE,"Require device tensors");
      if (engine->getTensorIOMode(name.c_str())==nvinfer1::TensorIOMode::kOUTPUT) {
        Array a; a.shape=ctx->getTensorShape(name.c_str()); std::size_t count=1;
        for (int d=0;d<a.shape.nbDims;++d) { check(a.shape.d[d]>0,"Unresolved output shape"); count*=a.shape.d[d]; }
        a.values.resize(count); arrays.emplace(name,std::move(a));
      }
      auto& a=arrays.at(name); auto& b=buffers[name]; cudaCheck(cudaMalloc(&b.p,a.values.size()*sizeof(float)));
      check(ctx->setTensorAddress(name.c_str(),b.p),"Cannot bind " + name);
      if (engine->getTensorIOMode(name.c_str())==nvinfer1::TensorIOMode::kINPUT)
        cudaCheck(cudaMemcpyAsync(b.p,a.values.data(),a.values.size()*sizeof(float),cudaMemcpyHostToDevice,stream.s));
    }
    check(ctx->enqueueV3(stream.s),"Inference failed");
    cudaCheck(cudaStreamSynchronize(stream.s));
    check(std::filesystem::create_directory(argv[4]),"Output directory must be new");
    for (int i=0;i<engine->getNbIOTensors();++i) {
      std::string name=engine->getIOTensorName(i);
      if (engine->getTensorIOMode(name.c_str())!=nvinfer1::TensorIOMode::kOUTPUT) continue;
      auto& a=arrays.at(name);
      cudaCheck(cudaMemcpy(a.values.data(),buffers.at(name).p,a.values.size()*sizeof(float),cudaMemcpyDeviceToHost));
      writeNpy(std::filesystem::path(argv[4])/(name+".npy"),a);
    }
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
