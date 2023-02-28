#include <sycl/sycl.hpp>
#include "common.h"
#include <cstdlib>


template <typename DataT,size_t LocalSize>
class LocalMemory{
    protected:
        size_t global_size;
        size_t local_size = LocalSize;

        std::vector<DataT> in_array1;
        std::vector<DataT> in_array2;

        std::vector<DataT> out_array1;

  
        PrefetchedBuffer<DataT, 1> in_buf1;
        PrefetchedBuffer<DataT, 1> in_buf2;    

        PrefetchedBuffer<DataT, 1> out_buf1;
        BenchmarkArgs& args;
    public:
        LocalMemory(BenchmarkArgs& _args) : args(_args) {}
        void setup(){
            global_size = args.problem_size;
            in_array1.resize(global_size);
            in_array2.resize(global_size);
            out_array1.resize(global_size);
            std::fill(in_array1.begin(), in_array1.end(), rand() % 1 + 1);
            std::fill(in_array2.begin(), in_array2.end(), rand() % 1 + 1);


            // buffer initialization
            in_buf1.initialize(args.device_queue, in_array1.data(), sycl::range<1>{global_size});
            in_buf2.initialize(args.device_queue, in_array2.data(), sycl::range<1>{global_size});

            out_buf1.initialize(args.device_queue, out_array1.data(), sycl::range<1>{global_size});
           


        }

        void run(std::vector<sycl::event>& events){
            events.push_back(
                args.device_queue.submit([&](sycl::handler& cgh) {
                    auto in_acc1 = in_buf1.template get_access<sycl::access_mode::read>(cgh);
                    auto in_acc2 = in_buf2.template get_access<sycl::access_mode::read>(cgh);

                    auto out_acc1 = out_buf1.template get_access<sycl::access_mode::write>(cgh);

                    sycl::local_accessor<DataT, 1> local_acc1{LocalSize, cgh};
                    sycl::local_accessor<DataT, 1> local_acc2{LocalSize, cgh};

                    
                    sycl::nd_range<1> ndr{global_size, local_size};
                    cgh.parallel_for<class LocalMemory>(ndr, [=](sycl::nd_item<1> item) {
                        sycl::id<1> lid = item.get_local_id();
                        sycl::id<1> gid = item.get_global_id();

                        local_acc1[lid] = in_acc1[gid];
                        local_acc2[lid] = in_acc2[gid];

                        #pragma unroll
                        for (size_t i = 0; i < LocalSize; i++) {
                            local_acc2[lid] *= local_acc1[lid];
                            local_acc2[lid] /= local_acc1[lid];
                            local_acc2[lid] += local_acc1[lid];
                        }

                        out_acc1[gid] = local_acc1[lid] + local_acc2[lid];                    });

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