#include "bitmap.h"
#include "common.h"
#include <iostream>
#include <sycl/sycl.hpp>

#define NUM_REP_KERNELS 8

enum FreqScalingApproach { PER_KERNEL, PHASE_AWARE, PER_APPLICATION };
namespace s = sycl;
class MMMM_SSSSKernel; // kernel forward declaration

/*
  This benchamrk is specifically built in order to create the best scenario in which apply the phase aware frequency
  change approach The benchmark use only two kernel with different energy characteristics: a matrix multiplication (M)
  and a sobel filter (S). These kernels are launched with linear dependencies creating the following SYCL task graph
  M->M->M->M->...->S->S->S->S->... The idea is to lauch this multi-kernel bench with three different approach:
  1. per application
  2. per kernel frequency scaling
  3. phase aware frequency scaling (we set the freq. two times one when the first M kernel is launched and the another
  time when we launch the S kernel)

  From the experimental evaluation we know that:
  1. Per kernel frequency for M is 1200 and for S is 495
  2. Phase aware to minimize the energy consumption we need two phase one with core_freq 1200 and the other with
     core_freq 495
  3. Per application we use as frequnecy the frequency at the middle between 495 and 1200 (855)
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
    // num itesrs for sobel is 2000
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

// TODO: change the frequency with respect to the type of approach
template <class T, FreqScalingApproach approach>
class MMMM_SSSS {
protected:
  size_t num_iters_mat_mul;
  size_t num_iters_sobel;
  std::vector<synergy::frequency> kernel_core_freqs;


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
  size_t size_mat_mul;
  size_t size_sobel;

  BenchmarkArgs args;

public:
  MMMM_SSSS(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    kernel_core_freqs.resize(NUM_REP_KERNELS*2);
    kernel_core_freqs[0]=1207;
    kernel_core_freqs[1]=1215;
    kernel_core_freqs[2]=1222;
    kernel_core_freqs[3]=1230;
    kernel_core_freqs[4]=495;
    kernel_core_freqs[5]=502;
    kernel_core_freqs[6]=510;
    kernel_core_freqs[7]=517;


    size_sobel = 3072;
    size_mat_mul = 1024;

    num_iters_sobel = 2;
    num_iters_mat_mul = 1;

    // mat_mul
    a.resize(size_mat_mul * size_mat_mul);
    b.resize(size_mat_mul * size_mat_mul);
    c.resize(size_mat_mul * size_mat_mul);

    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);

    a_buf.initialize(args.device_queue, a.data(), s::range<1>{size_mat_mul * size_mat_mul});
    b_buf.initialize(args.device_queue, b.data(), s::range<1>{size_mat_mul * size_mat_mul});
    c_buf.initialize(args.device_queue, c.data(), s::range<1>{size_mat_mul * size_mat_mul});

    // sobel
    input.resize(size_sobel * size_sobel);
    load_bitmap_mirrored("../Brommy.bmp", size_sobel, input);
    output.resize(size_sobel * size_sobel);

    input_buf.initialize(args.device_queue, input.data(), s::range<2>(size_sobel, size_sobel));
    output_buf.initialize(args.device_queue, output.data(), s::range<2>(size_sobel, size_sobel));
  }

  void run(std::vector<sycl::event>& events) {
    // for i=0 to 4 submit mat_mul. M->M->M->M ... ->S->S->S->S
    for(size_t i = 0; i < NUM_REP_KERNELS; i++) {
      if(i == 0 && approach == FreqScalingApproach::PHASE_AWARE) {
        events.push_back(args.device_queue.submit(0, 1117, [&](s::handler& cgh) {
          auto acc_a = a_buf.template get_access<s::access_mode::read>(cgh);
          auto acc_b = b_buf.template get_access<s::access_mode::read>(cgh);
          auto acc_c = c_buf.template get_access<s::access_mode::read_write>(cgh);
          cgh.parallel_for(s::range<2>{size_mat_mul, size_mat_mul},
              matrix_mul<T>(size_mat_mul, num_iters_mat_mul, acc_a, acc_b, acc_c)); // end parallel for
        }));
      } else if(approach == FreqScalingApproach::PER_KERNEL) {
        events.push_back(args.device_queue.submit(0, 1117, [&](s::handler& cgh) {
          auto acc_a = a_buf.template get_access<s::access_mode::read>(cgh);
          auto acc_b = b_buf.template get_access<s::access_mode::read>(cgh);
          auto acc_c = c_buf.template get_access<s::access_mode::read_write>(cgh);
          cgh.parallel_for(s::range<2>{size_mat_mul, size_mat_mul},
              matrix_mul<T>(size_mat_mul, num_iters_mat_mul, acc_a, acc_b, acc_c)); // end parallel for
        }));
      } else {
        events.push_back(args.device_queue.submit([&](s::handler& cgh) {
          auto acc_a = a_buf.template get_access<s::access_mode::read>(cgh);
          auto acc_b = b_buf.template get_access<s::access_mode::read>(cgh);
          auto acc_c = c_buf.template get_access<s::access_mode::read_write>(cgh);
          cgh.parallel_for(s::range<2>{size_mat_mul, size_mat_mul},
              matrix_mul<T>(size_mat_mul, num_iters_mat_mul, acc_a, acc_b, acc_c)); // end parallel for
        }));
      } // end events.push back
    }

    for(int i = 0; i < NUM_REP_KERNELS; i++) {
      if(i == 0 && approach == FreqScalingApproach::PHASE_AWARE) {
        events.push_back(args.device_queue.submit(0, 187, [&](s::handler& cgh) {
          auto in = input_buf.template get_access<s::access_mode::read>(cgh);
          auto out = output_buf.template get_access<s::access_mode::discard_write>(cgh);

          cgh.depends_on(events[events.size() - 1]);
          cgh.parallel_for(
              s::range<2>{size_sobel, size_sobel}, sobel(size_sobel, num_iters_sobel, in, out)); // end parallel for
        }));
      } else if(approach == FreqScalingApproach::PER_KERNEL) {
        events.push_back(args.device_queue.submit(0, 187, [&](s::handler& cgh) {
          auto in = input_buf.template get_access<s::access_mode::read>(cgh);
          auto out = output_buf.template get_access<s::access_mode::discard_write>(cgh);
          cgh.depends_on(events[events.size() - 1]);
          cgh.parallel_for(
              s::range<2>{size_sobel, size_sobel}, sobel(size_sobel, num_iters_sobel, in, out)); // end parallel for
        }));
      } else {
        events.push_back(args.device_queue.submit([&](s::handler& cgh) {
          auto in = input_buf.template get_access<s::access_mode::read>(cgh);
          auto out = output_buf.template get_access<s::access_mode::discard_write>(cgh);
          cgh.depends_on(events[events.size() - 1]);
          cgh.parallel_for(
              s::range<2>{size_sobel, size_sobel}, sobel(size_sobel, num_iters_sobel, in, out)); // end parallel for
        }));
      }
    }
  }


  bool verify(VerificationSetting& ver) {
    c_buf.reset();
    for(int i = 0; i < size_mat_mul * size_mat_mul; i++)
      if(size_mat_mul * NUM_REP_KERNELS != c[i])
        return false;


    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MMMM_SSSS_";
    name << ReadableTypename<T>::name;

    if constexpr(approach == FreqScalingApproach::PER_APPLICATION) {
      name << "_app";
    } else if constexpr(approach == FreqScalingApproach::PER_KERNEL) {
      name << "_kernel";
    } else if constexpr(approach == FreqScalingApproach::PHASE_AWARE) {
      name << "_phase";
    }
    return name.str();
  }
};


// run the benchmark with different approach
template <typename T, FreqScalingApproach approach>
void runFreqScalingApporach(BenchmarkApp& app) {
  app.run<MMMM_SSSS<T, approach>>();
}

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  runFreqScalingApporach<float, FreqScalingApproach::PER_KERNEL>(app);

  return 0;
}
