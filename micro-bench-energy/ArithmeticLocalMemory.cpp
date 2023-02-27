#include <cstdlib>
#include <sycl/sycl.hpp>
#include "common.h"


template<size_t LocalSize, size_t Coarsening, size_t PercFloatAddsub, size_t PercFloatMul, size_t PercFloatDiv, size_t PercIntAddsub, size_t PercIntMul, size_t PercIntDiv, size_t PercSpFunc>
class ArithmeticLocalMemory{
    protected:
        size_t global_size;
        size_t local_size = LocalSize;
        BenchmarkArgs& args;

        std::vector<float> in_float_array;
        std::vector<int> in_int_array;
        std::vector<float> out_array;
         
        PrefetchedBuffer<float, 1> in_float_buf;
        PrefetchedBuffer<int, 1> in_int_buf;    
        PrefetchedBuffer<float, 1> out_buf;
    
    public:
        ArithmeticLocalMemory(BenchmarkArgs& _args): args(_args){}

        void setup(){
            global_size = args.problem_size;
            in_float_array.resize(global_size);
            in_int_array.resize(global_size);
            out_array.resize(global_size);

            srand(1);
            float f_fill = rand() % 1 + 1;
            int i_fill = rand() % 1 + 1;

            std::fill(in_float_array.begin(), in_float_array.end(), f_fill);
            std::fill(in_int_array.begin(), in_int_array.end(), i_fill);

            in_float_buf.initialize(args.device_queue,in_float_array.data(), sycl::range<1>{global_size});
            in_int_buf.initialize(args.device_queue, in_int_array.data(), sycl::range<1>{global_size});
            out_buf.initialize(args.device_queue, out_array.data(), sycl::range<1>{global_size});
        }
        void run(std::vector<sycl::event>& events){
            events.push_back(
                args.device_queue.submit([&] (sycl::handler& cgh){
                    auto in_float_acc = in_float_buf.get_access<sycl::access_mode::read>(cgh);
                    auto in_int_acc = in_int_buf.get_access<sycl::access_mode::read>(cgh);
                    
                    // Local accessor
                    sycl::local_accessor<float, 1> in_float_local_acc{sycl::range<1>{LocalSize * Coarsening}, cgh};
                    sycl::local_accessor<int, 1> in_int_local_acc{sycl::range<1>{LocalSize * Coarsening}, cgh};

                    auto out_acc = out_buf.get_access<sycl::access_mode::write>(cgh);
                    
                    sycl::range<1> r{global_size / Coarsening};
                    sycl::range<1> local_r{LocalSize};

                    cgh.parallel_for<class ArithmeticLocalMemory>(sycl::nd_range<1>{r, local_r}, [=](sycl::nd_item<1> it) {
                        sycl::group<1> group = it.get_group();
                        sycl::id<1> global_id = it.get_global_id();
                        sycl::id<1> local_id = it.get_local_id();
                        size_t global_base_data_index = global_id.get(0) * Coarsening;
                        size_t local_base_data_index = local_id.get(0) * Coarsening;

                        // LocalSize * Coarsening
                        // clang-format off
                        #pragma unroll
                        for (size_t i = 0; i < Coarsening; i++) {
                        in_float_local_acc[local_base_data_index + i] = in_float_acc[global_base_data_index + i];
                        in_int_local_acc[local_base_data_index + i] = in_int_acc[global_base_data_index + i];
                        }

                        sycl::group_barrier(group);

                        // clang-format off
                        #pragma unroll
                        for (size_t i = 0; i < Coarsening; i++) {
                        size_t data_index = local_base_data_index + i;

                        float f0 = in_float_local_acc[data_index];
                        int i0 = in_int_local_acc[data_index];
                        int zero = i0 >> 1;

                        #pragma unroll
                        for (size_t j = 0; j < LocalSize * Coarsening; j++) {
                            float f1 = in_int_local_acc[j];
                            int i1 = in_float_local_acc[j];
                            
                            // clang-format off
                            #pragma unroll
                            for (size_t f_mul = 0; f_mul < PercFloatMul; f_mul++) {
                            f1 = f1 * f0;
                            f0 = f0 * f1;
                            }
                            
                            // clang-format off
                            #pragma unroll
                            for (size_t f_div = 0; f_div < PercFloatDiv; f_div++) {
                            f1 = f1 / f0;
                            f0 = f0 / f1;
                            }

                            // clang-format off
                            #pragma unroll
                            for (size_t f_sp = 0; f_sp < PercSpFunc; f_sp++) {
                            f1 = sycl::acos(f0);
                            f0 = f0 * f0 + f1;
                            }

                            // clang-format off
                            #pragma unroll
                            for (size_t f_addsub = 0; f_addsub < PercFloatAddsub; f_addsub++) {
                            f0 = f0 + f1;
                            f1 = f1 + f0;
                            }
                            
                            // clang-format off
                            #pragma unroll
                            for (size_t i_mul = 0; i_mul < PercIntMul; i_mul++) {
                            i1 = i1 * i0;
                            i0 = i0 * i1;
                            }
                            // i0 = i0 * i1;
                            
                            // clang-format off
                            #pragma unroll
                            for (size_t i_div = 0; i_div < PercIntDiv; i_div++) {
                            i1 = i1 / i0;
                            i0 = i0 / i1;
                            }

                            // clang-format off
                            #pragma unroll
                            for (size_t i_addsub = 0; i_addsub < PercIntAddsub; i_addsub++) {
                            i1 = i1 + i0;
                            i0 = i0 + i1;
                            }

                            f0 = f0 * zero + 1;
                            i0 = i0 * zero + 1;
                        }

                        out_acc[data_index] = i0 + f0;
                        }
                    });                    

                }) //end submit
            ); // end push back
        }

   bool verify(VerificationSetting& ver) {
    return true;
  }

  static std::string getBenchmarkName() { 
    std::string name = "ArithmeticLocalMemory_";
    name.append(std::to_string(LocalSize))
        .append("_")
        .append(std::to_string(Coarsening))
        .append("_")
        .append(std::to_string(PercFloatAddsub))
        .append("_")
        .append(std::to_string(PercFloatDiv))
        .append("_")
        .append(std::to_string(PercIntAddsub))
        .append("_")
        .append(std::to_string(PercIntMul))
        .append("_")
        .append(std::to_string(PercIntDiv))
        .append("_")
        .append(std::to_string(PercSpFunc));
    return name; 
  }

};



int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  
  // float
  app.run<ArithmeticLocalMemory< 8, 1, 2, 3, 3, 0, 0, 0, 0>>();
  app.run<ArithmeticLocalMemory< 16, 1, 2, 3, 3, 0, 0, 0, 0>>();
  app.run<ArithmeticLocalMemory< 32, 1, 2, 3, 3, 0, 0, 0, 0>>();

  app.run<ArithmeticLocalMemory< 8, 2, 2, 3, 3, 0, 0, 0, 0>>();
  app.run<ArithmeticLocalMemory< 16, 2, 2, 3, 3, 0, 0, 0, 0>>();
  app.run<ArithmeticLocalMemory< 32, 2, 2, 3, 3, 0, 0, 0, 0>>();

  
  // int
  app.run<ArithmeticLocalMemory< 8, 1, 0, 0, 0, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 16, 1, 0, 0, 0, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 32, 1, 0, 0, 0, 2, 3, 3, 0>>();

  app.run<ArithmeticLocalMemory< 8, 2, 0, 0, 0, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 16, 2, 0, 0, 0, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 32, 2, 0, 0, 0, 2, 3, 3, 0>>();

  
  // float + sp
  app.run<ArithmeticLocalMemory< 8, 1, 2, 3, 3, 0, 0, 0, 1>>();
  app.run<ArithmeticLocalMemory< 16, 1, 2, 3, 3, 0, 0, 0, 1>>();
  app.run<ArithmeticLocalMemory< 32, 1, 2, 3, 3, 0, 0, 0, 1>>();

  app.run<ArithmeticLocalMemory< 8, 2, 2, 3, 3, 0, 0, 0, 1>>();
  app.run<ArithmeticLocalMemory< 16, 2, 2, 3, 3, 0, 0, 0, 1>>();
  app.run<ArithmeticLocalMemory< 32, 2, 2, 3, 3, 0, 0, 0, 1>>();

  
  // int + sp
  app.run<ArithmeticLocalMemory< 8, 1, 0, 0, 0, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 16, 1, 0, 0, 0, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 32, 1, 0, 0, 0, 2, 3, 3, 1>>();

  app.run<ArithmeticLocalMemory< 8, 2, 0, 0, 0, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 16, 2, 0, 0, 0, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 32, 2, 0, 0, 0, 2, 3, 3, 1>>();

  
  // equal float and int
  app.run<ArithmeticLocalMemory< 8, 1, 2, 3, 3, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 16, 1, 2, 3, 3, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 32, 1, 2, 3, 3, 2, 3, 3, 0>>();


  app.run<ArithmeticLocalMemory< 8, 2, 2, 3, 3, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 16,2, 2, 3, 3, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 32,2, 2, 3, 3, 2, 3, 3, 0>>();

  app.run<ArithmeticLocalMemory< 8, 1, 2, 3, 3, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 16, 1, 2, 3, 3, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 32, 1, 2, 3, 3, 2, 3, 3, 1>>();

  app.run<ArithmeticLocalMemory< 8, 2, 2, 3, 3, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 16, 2, 2, 3, 3, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 32, 2, 2, 3, 3, 2, 3, 3, 1>>();

  
  // more float than int
  app.run<ArithmeticLocalMemory< 8, 1, 4, 5, 5, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 16, 1, 4, 5, 5, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 32, 1, 4, 5, 5, 2, 3, 3, 0>>();

  app.run<ArithmeticLocalMemory< 8, 2, 4, 5, 5, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 16, 2, 4, 5, 5, 2, 3, 3, 0>>();
  app.run<ArithmeticLocalMemory< 32, 2, 4, 5, 5, 2, 3, 3, 0>>();

  app.run<ArithmeticLocalMemory< 8, 1, 4, 5, 5, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 16, 1, 4, 5, 5, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 32, 1, 4, 5, 5, 2, 3, 3, 1>>();

  app.run<ArithmeticLocalMemory< 8, 2, 4, 5, 5, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 16, 2, 4, 5, 5, 2, 3, 3, 1>>();
  app.run<ArithmeticLocalMemory< 32, 2, 4, 5, 5, 2, 3, 3, 1>>();

  
  // more int than float 
  app.run<ArithmeticLocalMemory< 8,1, 2, 3, 3, 4, 5, 5, 0>>();
  app.run<ArithmeticLocalMemory< 16,1, 2, 3, 3, 4, 5, 5, 0>>();
  app.run<ArithmeticLocalMemory< 32,1, 2, 3, 3, 4, 5, 5, 0>>();

  app.run<ArithmeticLocalMemory< 8,2, 2, 3, 3, 4, 5, 5, 0>>();
  app.run<ArithmeticLocalMemory< 16,2, 2, 3, 3, 4, 5, 5, 0>>();
  app.run<ArithmeticLocalMemory< 32,2, 2, 3, 3, 4, 5, 5, 0>>();

  app.run<ArithmeticLocalMemory< 8,1, 2, 3, 3, 4, 5, 5, 1>>();
  app.run<ArithmeticLocalMemory< 16,1, 2, 3, 3, 4, 5, 5, 1>>();
  app.run<ArithmeticLocalMemory< 32,1, 2, 3, 3, 4, 5, 5, 1>>();

  app.run<ArithmeticLocalMemory< 8, 2, 2, 3, 3, 4, 5, 5, 1>>();
  app.run<ArithmeticLocalMemory< 16, 2, 2, 3, 3, 4, 5, 5, 1>>();
  app.run<ArithmeticLocalMemory< 32, 2, 2, 3, 3, 4, 5, 5, 1>>();


  return 0;
}