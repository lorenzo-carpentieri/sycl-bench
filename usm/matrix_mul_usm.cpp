#include <iostream>
#include <sycl/sycl.hpp>

#include "bitmap.h"
#include "common.h"


namespace s = sycl;
class MatrixMulKernel; // kernel forward declaration

template<class T>
class matrixMul{
    private:
        int num_iters;
        int size;
        const T* in_A;
        const T* in_B;
        T* out;

    public: 
        matrixMul(
            int num_iters,
            int size, 
            const T* in_A,
            const T* in_B,
            T* out   
        ):
        num_iters(num_iters),
        size(size),
        in_A(in_A),
        in_B(in_B), 
        out(out)
        {}

        void operator()(s::id<2> gid)const {
            int gidx = gid.get(0);
            int gidy = gid.get(1);
            for(size_t i = 0; i < num_iters; i++) {
              for(int k = 0; k < size; k++)
                  out[gidx*size+gidy] += in_A[gidx*size+k] * in_B[k*size+gidy];
            }
        }
};

template<class T>
class MatrixMulUSM {
protected:
  size_t num_iters;
  std::vector<T> a;
  std::vector<T> b;
  std::vector<T> c;
  T* dev_a;
  T* dev_b;
  T* dev_c;

    
  size_t size;
  BenchmarkArgs args;

public:
  MatrixMulUSM(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    num_iters = args.num_iterations;
    a.resize(size*size);
    b.resize(size*size);
    c.resize(size*size);

    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);
    // allocate memory 
    dev_a   = s::malloc_device<T>(size*size, args.device_queue);
    dev_b   = s::malloc_device<T>(size*size, args.device_queue);
    dev_c = s::malloc_device<T>(size*size, args.device_queue);

    //init device 
    args.device_queue.memcpy(dev_a, &a[0], (size*size)*sizeof(T)); 
    args.device_queue.memcpy(dev_b, &b[0], (size*size)*sizeof(T));   
    args.device_queue.memcpy(dev_c, &c[0], (size*size)*sizeof(T)); 
    
    args.device_queue.wait();

  }

  void run(std::vector<sycl::event>& events) {
     // allocate memory 
    dev_a   = s::malloc_device<T>(size*size, args.device_queue);
    dev_b   = s::malloc_device<T>(size*size, args.device_queue);
    dev_c = s::malloc_device<T>(size*size, args.device_queue);

    //init device 
    args.device_queue.memcpy(dev_a, &a[0], (size*size)*sizeof(T)); 
    args.device_queue.memcpy(dev_b, &b[0], (size*size)*sizeof(T));   
    args.device_queue.memcpy(dev_c, &c[0], (size*size)*sizeof(T)); 
    
    args.device_queue.wait();
    
    events.push_back(
        args.device_queue.submit([&](s::handler &cgh){
            cgh.parallel_for(s::range<2>{size, size}, matrixMul<T>(num_iters, size, dev_a, dev_b, dev_c));//end parallel for
        })
    );// end events.push back
        
    args.device_queue.wait();
    args.device_queue.memcpy(&c[0], dev_c, (size*size)*sizeof(T));
    args.device_queue.wait();
      
    s::free(dev_a, args.device_queue);
    s::free(dev_b, args.device_queue);
    s::free(dev_c, args.device_queue); 
  }


  bool verify(VerificationSetting& ver) {
    for(int i = 0; i < size*size; i++)
      if(num_iters*size != c[i])
        return false;
          
    
    return true;
  }

  static std::string getBenchmarkName() { return "Matrix_mul_usm"; }

}; 


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<MatrixMulUSM<int>>();
  return 0;
}
