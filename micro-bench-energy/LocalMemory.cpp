#include <sycl/sycl.hpp>
#include "common.h"


template <typename DataT,size_t LocalSize>
class LocalMemory{
    protected:
        size_t global_size;
        size_t local_size = LocalSize;

        std::vector<DataT> in_array;
        std::vector<DataT> out_array;
  
        PrefetchedBuffer<DataT, 1> in_buf;    
        PrefetchedBuffer<DataT, 1> out_buf;
        BenchmarkArgs& args;
    public:
        LocalMemory(BenchmarkArgs& _args) : args(_args) {}
        void setup(){
            global_size = args.problem_size;
            in_array.resize(global_size);
            out_array.resize(global_size);
            std::fill(in_array.begin(), in_array.end(), 1);

            // buffer initialization
            in_buf.initialize(args.device_queue, in_array.data(), sycl::range<1>{global_size});
            out_buf.initialize(args.device_queue, out_array.data(), sycl::range<1>{global_size});
           


        }

        void run(std::vector<sycl::event>& events){
            events.push_back(
                args.device_queue.submit([&](sycl::handler& cgh) {
                    auto in_acc = in_buf.template get_access<sycl::access_mode::read>(cgh);
                    auto out_acc = out_buf.template get_access<sycl::access_mode::write>(cgh);
                    sycl::local_accessor<DataT, 1> local_acc{LocalSize, cgh};
                    
                    sycl::nd_range<1> ndr{global_size, local_size};
                    cgh.parallel_for<class LocalMemory>(ndr, [=](sycl::nd_item<1> item) {
                        sycl::id<1> lid = item.get_local_id();
                        sycl::id<1> gid = item.get_global_id();

                        local_acc[lid] = in_acc[gid];
                        out_acc[gid] = local_acc[lid];
                    });

                })// end submit
            );  // end push back
        }

  static std::string getBenchmarkName() { 
    std::string name = "LocalMemory_";
    name.append(std::is_same_v<DataT, int>? "int":"float")
        .append("_")
        .append(std::to_string(LocalSize));
    return name; 
  }

  bool verify(VerificationSetting& ver) {
    out_buf.reset();
    DataT sum = 0;
    for (DataT value : out_array) {
        if(value != 1)
            return false;
        sum += value;
    }
    return true;
  }

};

int main(int argc, char** argv) {
  
  BenchmarkApp app(argc, argv);
  // int
  app.run<LocalMemory<int, 8>>();
  app.run<LocalMemory<int, 16>>();
  app.run<LocalMemory<int, 32>>();
  app.run<LocalMemory<int, 64>>();
  
  // float
  app.run<LocalMemory<float, 8>>();
  app.run<LocalMemory<float, 16>>();
  app.run<LocalMemory<float, 32>>();
  app.run<LocalMemory<float, 64>>();

  return 0;
}