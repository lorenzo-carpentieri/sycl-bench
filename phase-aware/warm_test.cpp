#include "bitmap.h"
#include "common.h"
#include <iostream>
#include <sycl/sycl.hpp>

#define NUM_REP_KERNELS 4

namespace s = sycl;
class WarmKernel; // kernel forward declaration

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
class vec_add {
private:
  int size;
  int num_iters;
  const s::accessor<T, 1, s::access_mode::read> in_A;
  const s::accessor<T, 1, s::access_mode::read> in_B;
  s::accessor<T, 1, s::access_mode::read_write> out;

public:
  vec_add(int size, int num_iters, const s::accessor<T, 1, s::access_mode::read> in_A,
      const s::accessor<T, 1, s::access_mode::read> in_B, s::accessor<T, 1, s::access_mode::read_write> out)
      : size(size), num_iters(num_iters), in_A(in_A), in_B(in_B), out(out) {}

  void operator()(s::id<1> gid) const {
    for(int i = 0; i < num_iters; i++)
        out[gid] = in_A[gid] + in_B[gid];
  }
};

template <class T>
class Warm {
protected:
  size_t num_iters;
  size_t size;

  // mat_mul input
  std::vector<T> a;
  std::vector<T> b;
  std::vector<T> c;

  PrefetchedBuffer<T, 1> a_buf;
  PrefetchedBuffer<T, 1> b_buf;
  PrefetchedBuffer<T, 1> c_buf;

  BenchmarkArgs args;

public:
  Warm(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;
    num_iters = args.num_iterations;
    // mat_mul
    a.resize(size);
    b.resize(size);
    c.resize(size);

    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);

    a_buf.initialize(args.device_queue, a.data(), s::range<1>{size});
    b_buf.initialize(args.device_queue, b.data(), s::range<1>{size});
    c_buf.initialize(args.device_queue, c.data(), s::range<1>{size});
  }

  void run(std::vector<sycl::event>& events) {

        events.push_back(args.device_queue.submit([&](s::handler& cgh) {
          auto acc_a = a_buf.template get_access<s::access_mode::read>(cgh);
          auto acc_b = b_buf.template get_access<s::access_mode::read>(cgh);
          auto acc_c = c_buf.template get_access<s::access_mode::read_write>(cgh);
          cgh.parallel_for(s::range<1>{size},
              vec_add<T>(size, num_iters, acc_a, acc_b, acc_c)); // end parallel for
        }));
       // end events.push back
  }


  bool verify(VerificationSetting& ver) {
    c_buf.reset();

    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "Warm_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};




int main(int argc, char** argv) {
  for (int  i=0; i < 10;i++){
    BenchmarkApp app(argc, argv);
    app.run<Warm<float>>();
  }
  
  return 0;
}
