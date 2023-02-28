#include "common.h"
#include <iostream>
#include <sycl/sycl.hpp>

template <size_t Coarsening, size_t PercFloatAddsub, size_t PercFloatMul, size_t PercFloatDiv, size_t PercIntAddsub, size_t PercIntMul, size_t PercIntDiv, size_t PercSpFunc>
class ArithmeticBench{
protected:
  std::vector<float> in_float_array;
  std::vector<int> in_int_array;
  std::vector<float> out_array;
  
  size_t size;
  
  PrefetchedBuffer<float, 1> in_float_buf;
  PrefetchedBuffer<int, 1> in_int_buf;    
  PrefetchedBuffer<float, 1> out_buf;

  BenchmarkArgs& args;

public:
  ArithmeticBench(BenchmarkArgs &_args): args(_args){}

  void setup() {
    // Set input and output size
    size = args.problem_size;
    in_float_array.resize(size);
    in_int_array.resize(size);
    out_array.resize(size);
    // Input intialization with trick to avoid compiler optimization. 
    srand(1);
    float f_fill = rand() % 1 + 1;
    int i_fill = rand() % 1 + 1;
    std::fill(in_float_array.begin(), in_float_array.end(), f_fill);
    std::fill(in_int_array.begin(), in_int_array.end(), i_fill);

    // Buffer init
    in_float_buf.initialize(args.device_queue,in_float_array.data(), sycl::range<1>{size});
    in_int_buf.initialize(args.device_queue, in_int_array.data(), sycl::range<1>{size});
    out_buf.initialize(args.device_queue, out_array.data(), sycl::range<1>{size});
    


  }

  void run(std::vector<sycl::event>& events){
    events.push_back(
      args.device_queue.submit(
        [&](sycl::handler& cgh){
          
          auto in_float_acc = in_float_buf.get_access<sycl::access_mode::read>(cgh);
          auto in_int_acc = in_int_buf.get_access<sycl::access_mode::read>(cgh);
          auto out_acc = out_buf.get_access<sycl::access_mode::write>(cgh);
          sycl::range<1> r{size / Coarsening};

          cgh.parallel_for(r, [=, _size=size](sycl::id<1> id) {
            size_t base_data_index = id.get(0) * Coarsening;

            // clang-format off
            #pragma unroll
            for (size_t i = 0; i < Coarsening; i++) {
              size_t data_index = base_data_index + i;

              float f0 = in_float_acc[data_index];
              int i0  = in_int_acc[data_index];
              
              float f1 = in_int_acc[(data_index + _size / 2) % _size];
              int i1 = in_float_acc[(data_index + _size / 2) % _size];
              
              // clang-format off
              #pragma unroll
              for (size_t f_mul = 0; f_mul < PercFloatMul; f_mul++) {
                f1 = f1 * f0;
                f0 = f0 * f1;
              }
              out_acc[data_index] *= f0;
              
              // clang-format off
              #pragma unroll
              for (size_t f_div = 0; f_div < PercFloatDiv; f_div++) {
                f1 = f1 / f0;
                f0 = f0 / f1;
              }
              out_acc[data_index] *= f0;

              // clang-format off
              #pragma unroll
              for (size_t f_sp = 0; f_sp < PercSpFunc; f_sp++) {
                f1 = sycl::acos(f0);
                f0 = f0 * f0 + f1;
              }
              out_acc[data_index] *= f0;

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
              i0 = i0 * i1;
              out_acc[data_index] *= i0;
              
              // clang-format off
              #pragma unroll
              for (size_t i_div = 0; i_div < PercIntDiv + 1; i_div++) {
                i1 = i1 / i0;
                i0 = i0 / i1;
              }
              out_acc[data_index] *= i0;

              // clang-format off
              #pragma unroll
              for (size_t i_addsub = 0; i_addsub < PercIntAddsub; i_addsub++) {
                i1 = i1 + i0;
                i0 = i0 + i1;
              }
              
              out_acc[data_index] *= (i0 + f0);
            }
          });//end parallel for
        }
      )
    );
  }

   bool verify(VerificationSetting& ver) {
    return true;
  }

  static std::string getBenchmarkName() { 
    std::string name = "Arithmetic_";
    name.append(std::to_string(Coarsening))
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

int main(int argc, char** argv)
{
    BenchmarkApp app(argc, argv);
    //float
    app.run<ArithmeticBench<1, 4, 5, 5, 0, 0, 0, 0>>();  
    app.run<ArithmeticBench<2, 4, 5, 5, 0, 0, 0, 0>>();  
  
    // int
    app.run<ArithmeticBench<1, 0, 0, 0, 4, 5, 5, 0>>();  
    app.run<ArithmeticBench<2, 0, 0, 0, 4, 5, 5, 0>>(); 

    // float + sp
    app.run<ArithmeticBench<1, 4, 5, 5, 0, 0, 0, 2>>();  
    app.run<ArithmeticBench<2, 4, 5, 5, 0, 0, 0, 2>>();  

    // int + sp
    app.run<ArithmeticBench<1, 0, 0, 0, 4, 5, 5, 2>>();   
    app.run<ArithmeticBench<2, 0, 0, 0, 4, 5, 5, 2>>();  
   
    // equal float and int
    app.run<ArithmeticBench<1, 4, 5, 5, 4, 5, 5, 0>>();   
    app.run<ArithmeticBench<2, 4, 5, 5, 4, 5, 5, 0>>();  
    app.run<ArithmeticBench<1, 4, 5, 5, 4, 5, 5, 2>>();   
    app.run<ArithmeticBench<2, 4, 5, 5, 4, 5, 5, 2>>();  
    
    // more float than int
    app.run<ArithmeticBench<1, 6, 7, 7, 4, 5, 5, 0>>();   
    app.run<ArithmeticBench<2, 6, 7, 7, 4, 5, 5, 0>>();  
    app.run<ArithmeticBench<1, 6, 7, 7, 4, 5, 5, 2>>();   
    app.run<ArithmeticBench<2, 6, 7, 7, 4, 5, 5, 2>>();  

    // more int than float
    app.run<ArithmeticBench<1, 4, 5, 5, 6, 7, 7, 0>>();   
    app.run<ArithmeticBench<2, 4, 5, 5, 6, 7, 7, 0>>();  
    app.run<ArithmeticBench<1, 4, 5, 5, 6, 7, 7, 2>>();   
    app.run<ArithmeticBench<2, 4, 5, 5, 6, 7, 7, 2>>();  

    return 0;
}