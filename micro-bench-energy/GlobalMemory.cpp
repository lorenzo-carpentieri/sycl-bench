#include <sycl/sycl.hpp>
#include "common.h"



template <typename DataT, size_t Coarsening>
class GlobaMemoryBench {
protected:
  std::vector<DataT> in_array;
  std::vector<DataT> out_array;
  
  PrefetchedBuffer<DataT, 1> in_buf;    
  PrefetchedBuffer<DataT, 1> out_buf;
  size_t size;
  BenchmarkArgs& args;

public:
  GlobaMemoryBench(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;
    in_array.resize(size);
    out_array.resize(size);

    std::fill(in_array.begin(), in_array.end(), 1);

    // buffer initialization
    in_buf.initialize(args.device_queue, in_array.data(), sycl::range<1>{size});
    out_buf.initialize(args.device_queue, out_array.data(), sycl::range<1>{size});


  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      auto in_acc = in_buf.template get_access<sycl::access_mode::read>(cgh);
      auto out_acc = out_buf.template get_access<sycl::access_mode::write>(cgh);
      sycl::range<1> r{size / Coarsening};

      cgh.parallel_for(r, [=](sycl::id<1> id) {
        size_t base_data_index = id.get(0) * Coarsening;

        #pragma unroll
        for(size_t i = 0; i < Coarsening; i++) {
          size_t data_index = base_data_index + i;
          out_acc[data_index] = in_acc[data_index];
        }
      });// end parallel_for
    }));  // end push back
  }

  static std::string getBenchmarkName() { 
    std::string name = "GlobalMemory_";
    name.append(std::is_same_v<DataT, int>? "int":"float")
        .append("_")
        .append(std::to_string(Coarsening));
    return name; 
  }

  bool verify(VerificationSetting& ver) {
    return true;
  }

};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  
  // int
  app.run<GlobaMemoryBench<int, 1>>();
  app.run<GlobaMemoryBench<int, 2>>();
  app.run<GlobaMemoryBench<int, 4>>();
  app.run<GlobaMemoryBench<int, 8>>();
  
  // float
  app.run<GlobaMemoryBench<float, 1>>();
  app.run<GlobaMemoryBench<float, 2>>();
  app.run<GlobaMemoryBench<float, 4>>();
  app.run<GlobaMemoryBench<float, 8>>();
 

  return 0;
}