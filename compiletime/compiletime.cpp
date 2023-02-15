// Skeleton for compile time measurements -- doesn't do anything on its own, but should compile successfully

#include <sycl/sycl.hpp>

namespace s = sycl;

#include <kernel_declarations.inc>

void run(size_t rt_size) {
  sycl::queue device_queue;
  #include <kernels.inc>
  device_queue.wait_and_throw();
}

int main(int argc, char** argv) {
  run(static_cast<size_t>(argc));
  // not intended to be run
  return -1;
}
