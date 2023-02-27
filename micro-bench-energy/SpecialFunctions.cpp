#include <sycl/sycl.hpp>
#include "common.h"


template <size_t Coarsening, size_t Iterations>
class SpecialFunctions{
    protected:
        std::vector<float> in_array;
        std::vector<float> out_array;
  
        PrefetchedBuffer<float, 1> in_buf;    
        PrefetchedBuffer<float, 1> out_buf;

        size_t size;
        BenchmarkArgs& args;

    public:
        SpecialFunctions(BenchmarkArgs& _args):args(_args){}

        void setup(){
            size = args.problem_size;
            in_array.resize(size);
            out_array.resize(size);

            std::fill(in_array.begin(), in_array.end(), 0.0f);
             
            // buffer initialization
            in_buf.initialize(args.device_queue, in_array.data(), sycl::range<1>{size});
            out_buf.initialize(args.device_queue, out_array.data(), sycl::range<1>{size});
        }

        void run(std::vector<sycl::event>& events){
            events.push_back(
                args.device_queue.submit([&](sycl::handler& cgh){
                    auto in_acc = in_buf.get_access<sycl::access_mode::read>(cgh);
                    auto out_acc = out_buf.get_access<sycl::access_mode::write>(cgh);
                    sycl::range<1> r{size / Coarsening};

                    cgh.parallel_for(r, [=](sycl::id<1> id) {
                        size_t base_data_index = id.get(0) * Coarsening;

                        #pragma unroll
                        for (size_t i = 0; i < Coarsening; i++) {
                        size_t data_index = base_data_index + i;

                        float f0, f1, f2;

                        f0 = in_acc[data_index];
                        f1 = f2 = f0;

                        #pragma unroll
                        for (size_t j = 0; j < Iterations; j++) {
                            f0 = sycl::cos(f1);
                            f1 = sycl::sin(f2);
                            f2 = sycl::tan(f1);
                        }

                        out_acc[data_index] = f0;
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
        out_buf.reset();

        for (float value : out_array) {
            if(value != 1)
                return false;
        }
        return true;
    }   

};

int main(int argc, char** argv) {
  
  BenchmarkApp app(argc, argv);
  app.run<SpecialFunctions<1, 1>>();
  app.run<SpecialFunctions<1, 64>>();
  app.run<SpecialFunctions<2, 1>>();
  app.run<SpecialFunctions<2, 64>>();
  app.run<SpecialFunctions<4, 1>>();
  app.run<SpecialFunctions<4, 64>>();
  app.run<SpecialFunctions<8, 1>>();
  app.run<SpecialFunctions<8, 64>>();

  return 0;
}