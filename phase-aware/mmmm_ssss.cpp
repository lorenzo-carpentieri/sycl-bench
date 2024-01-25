#include <iostream>
#include <sycl/sycl.hpp>
#include "common.h"
#include "bitmap.h"

#define NUM_REP_KERNELS 4

namespace s = sycl;
class MMMM_SSSSKernel; // kernel forward declaration

/*
  This benchamrk is specifically built in order to create the best scenario in which apply the phase aware frequency change approach
  The benchmark use only two kernel with different energy characteristics: a matrix multiplication (M) and a sobel filter (S).
  These kernels are launched with linear dependencies creating the following SYCL task graph M->M->M->M->...->S->S->S->S->...
  The idea is to lauch this multi-kernel bench with three different approach:
  1. per application 
  2. per kernel frequency scaling
  3. phase aware frequency scaling (we set the freq. two times one when the first M kernel is launched and the another time when we launch the S kernel) 
*/

// matrix mul
template <class T>
class matrix_mul {
private:
  int size;
  int num_iters;
  const s::accessor<T, 1, s::access_mode::read> in_A;
  const s::accessor<T, 1, s::access_mode::read> in_B;
  s::accessor<T, 1, s::access_mode::read_write> out;

public:
  matrix_mul(int size, int num_iters, const s::accessor<T, 1, s::access_mode::read> in_A,
      const s::accessor<T, 1, s::access_mode::read> in_B, s::accessor<T, 1, s::access_mode::read_write> out)
      : size(size), num_iters(num_iters), in_A(in_A), in_B(in_B), out(out) {}

  void operator()(s::id<2> gid) const {
    int gidx = gid.get(0);
    int gidy = gid.get(1);
    for(int iter = 0; iter < num_iters; iter++)
      for(int k = 0; k < size; k++) out[gidx * size + gidy] += in_A[gidx * size + k] * in_B[k * size + gidy];
  }
};

// Sobel 3
class sobel {
private:
  int size;
  int num_iters;
  const s::accessor<s::float4, 2, s::access_mode::read> in;
  s::accessor<s::float4, 2, s::access_mode::discard_write> out;

public:
  sobel(int size, int num_iters, const s::accessor<s::float4, 2, s::access_mode::read> in,
      s::accessor<s::float4, 2, s::access_mode::discard_write> out)
      : size(size), num_iters(num_iters), in(in), out(out) {}

  void operator()(s::id<2> gid) const {
    const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int x = gid[0];
    int y = gid[1];

    for(size_t i = 0; i < num_iters; i++) {
      sycl::float4 Gx = sycl::float4(0, 0, 0, 0);
      sycl::float4 Gy = sycl::float4(0, 0, 0, 0);
      const int radius = 3;

      // constant-size loops in [0,1,2]
      for(int x_shift = 0; x_shift < 3; x_shift++) {
        for(int y_shift = 0; y_shift < 3; y_shift++) {
          // sample position
          uint xs = x + x_shift - 1; // [x-1,x,x+1]
          uint ys = y + y_shift - 1; // [y-1,y,y+1]
          // for the same pixel, convolution is always 0
          if(x == xs && y == ys)
            continue;
          // boundary check
          if(xs < 0 || xs >= size || ys < 0 || ys >= size)
            continue;

          // sample color
          sycl::float4 sample = in[{xs, ys}];

          // convolution calculation
          int offset_x = x_shift + y_shift * radius;
          int offset_y = y_shift + x_shift * radius;

          float conv_x = kernel[offset_x];
          sycl::float4 conv4_x = sycl::float4(conv_x);
          Gx += conv4_x * sample;

          float conv_y = kernel[offset_y];
          sycl::float4 conv4_y = sycl::float4(conv_y);
          Gy += conv4_y * sample;
        }
      }
      // taking root of sums of squares of Gx and Gy
      sycl::float4 color = hypot(Gx, Gy);
      sycl::float4 minval = sycl::float4(0.0, 0.0, 0.0, 0.0);
      sycl::float4 maxval = sycl::float4(1.0, 1.0, 1.0, 1.0);
      out[gid] = clamp(color, minval, maxval);
    }
  }
};


template <class T>
class MMMM_SSSS {
protected:
  size_t num_iters;

  // mat_mul input
  std::vector<T> a;
  std::vector<T> b;
  std::vector<T> c;

  PrefetchedBuffer<T, 1> a_buf;
  PrefetchedBuffer<T, 1> b_buf;
  PrefetchedBuffer<T, 1> c_buf;

  // sobel input
  std::vector<sycl::float4> input;
  std::vector<sycl::float4> output;

  size_t w, h; // size of the input picture
  PrefetchedBuffer<sycl::float4, 2> input_buf;
  PrefetchedBuffer<sycl::float4, 2> output_buf;
  size_t size;
  BenchmarkArgs args;

public:
  MMMM_SSSS(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    num_iters = args.num_iterations;
    
    // mat_mul
    a.resize(size * size);
    b.resize(size * size);
    c.resize(size * size);

    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);

    a_buf.initialize(args.device_queue, a.data(), s::range<1>{size * size});
    b_buf.initialize(args.device_queue, b.data(), s::range<1>{size * size});
    c_buf.initialize(args.device_queue, c.data(), s::range<1>{size * size});
  
    //sobel
    input.resize(size * size);
    load_bitmap_mirrored("../Brommy.bmp", size, input);
    output.resize(size * size);

    input_buf.initialize(args.device_queue, input.data(), s::range<2>(size, size));
    output_buf.initialize(args.device_queue, output.data(), s::range<2>(size, size));
  }

  void run(std::vector<sycl::event>& events) {
    // for i=0 to 4 submit mat_mul. M->M->M->M ... ->S->S->S->S
    
    for(size_t i = 0; i < NUM_REP_KERNELS; i++) {
      events.push_back(args.device_queue.submit([&](s::handler& cgh) {
        auto acc_a = a_buf.template get_access<s::access_mode::read>(cgh);
        auto acc_b = b_buf.template get_access<s::access_mode::read>(cgh);
        auto acc_c = c_buf.template get_access<s::access_mode::read_write>(cgh);
        cgh.parallel_for(
            s::range<2>{size, size}, matrix_mul<T>(size, num_iters, acc_a, acc_b, acc_c)); // end parallel for
      }));                                                                                 // end events.push back
    }

    for(int i = 0; i < NUM_REP_KERNELS; i++) {
      events.push_back(args.device_queue.submit([&](s::handler& cgh) {
        auto in = input_buf.template get_access<s::access_mode::read>(cgh);
        auto out = output_buf.template get_access<s::access_mode::discard_write>(cgh);
        if(i==0) 
          cgh.depends_on(events[events.size()-1]);
        cgh.parallel_for(s::range<2>{size, size},sobel(size, num_iters, in, out)); // end parallel for
      }));
    }
  }


  bool verify(VerificationSetting& ver) {
    c_buf.reset();
    for(int i = 0; i < size * size; i++)
      if(size * NUM_REP_KERNELS != c[i])
        return false;


    return true;
  }

  static std::string getBenchmarkName() { return "MMMM_SSSS"; }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<MMMM_SSSS<float>>();
  return 0;
}
