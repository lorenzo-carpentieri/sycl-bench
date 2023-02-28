#include <sycl/sycl.hpp>
#include "common.h"


template <size_t Coarsening, size_t Iterations>
class SpecialFunctions{
    protected:
        std::vector<float> in_array;
        std::vector<float> out_array1;
        std::vector<float> out_array2;

  
        PrefetchedBuffer<float, 1> in_buf;    
        PrefetchedBuffer<float, 1> out_buf1;
        PrefetchedBuffer<float, 1> out_buf2;


        size_t size;
        BenchmarkArgs& args;

    public:
        SpecialFunctions(BenchmarkArgs& _args):args(_args){}

        void setup(){
            size = args.problem_size;
            in_array.resize(size);
            out_array1.resize(size);
            out_array2.resize(size);

            std::fill(in_array.begin(), in_array.end(), 0.0f);
             
            // buffer initialization
            in_buf.initialize(args.device_queue, in_array.data(), sycl::range<1>{size});
            out_buf1.initialize(args.device_queue, out_array1.data(), sycl::range<1>{size});
            out_buf2.initialize(args.device_queue, out_array2.data(), sycl::range<1>{size});

        }

        void run(std::vector<sycl::event>& events){
            events.push_back(
                args.device_queue.submit([&](sycl::handler& cgh){
                    auto in_acc = in_buf.get_access<sycl::access_mode::read>(cgh);
                    auto out_acc1 = out_buf1.get_access<sycl::access_mode::write>(cgh);
                    auto out_acc2 = out_buf2.get_access<sycl::access_mode::write>(cgh);

                    sycl::range<1> r{size / Coarsening};

                    cgh.parallel_for(r, [=](sycl::id<1> id) {
                        size_t base_data_index = id.get(0) * Coarsening;

                        #pragma unroll
                        for (size_t i = 0; i < Coarsening; i++) {
                            size_t data_index = base_data_index + i;

                            float f0, f1, f2;

                            f0 = in_acc[data_index];
                            f1 = f2 = f0 = in_acc[data_index + 1];

                            #pragma unroll
                            for (size_t j = 0; j < Iterations; j++) {
                                out_acc2[data_index] = sycl::cos(out_acc2[data_index]);
                                f0 = sycl::sin(f2);
                                f2 = sycl::tan(f0);
                            }

                            out_acc1[data_index] = f2; 
                        }                       
                    });                    

                }) // end submit
            );
        }

    static std::string getBenchmarkName() { 
        std::string name = "SpecialFunctions_";
        name.append(std::to_string(Coarsening))
            .append("_")
            .append(std::to_string(Iterations));
        return name; 
    }

    bool verify(VerificationSetting& ver) {
       return true;
    }   

};

int main(int argc, char** argv) {
  
  BenchmarkApp app(argc, argv);
  app.run<SpecialFunctions<1, 1>>();
  app.run<SpecialFunctions<1, 8>>();
  app.run<SpecialFunctions<1, 16>>();



  app.run<SpecialFunctions<2, 1>>();
  app.run<SpecialFunctions<2, 8>>();
  app.run<SpecialFunctions<2, 16>>();


  app.run<SpecialFunctions<4, 1>>();
  app.run<SpecialFunctions<4, 8>>();
  app.run<SpecialFunctions<4, 16>>();


  app.run<SpecialFunctions<8, 1>>();
  app.run<SpecialFunctions<8, 8>>();
  app.run<SpecialFunctions<8, 16>>();


  return 0;
}