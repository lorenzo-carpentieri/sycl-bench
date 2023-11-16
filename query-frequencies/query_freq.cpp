#include <iostream>
#include <synergy.hpp>
#include <string.h>

int gpu_backend_selector(const sycl::device &d)
{
  auto vendor_name = d.get_platform().get_info<sycl::info::platform::name>();
  if (vendor_name.find("HIP") != std::string::npos || vendor_name.find("CUDA") != std::string::npos || vendor_name.find("Level") != std::string::npos)
    return 100;
  return 0;
}

int main(int argc, char *argv[])
{
  synergy::queue q{gpu_backend_selector};

  auto device = q.get_synergy_device();
  auto mem_freq = device.supported_uncore_frequencies();
  auto core_freq = device.supported_core_frequencies();

  if (strcmp(argv[1], "memory") == 0)
    for (auto freq : mem_freq)
      std::cout << freq << std::endl;
  if (strcmp(argv[1], "core") == 0)
    for (auto freq : core_freq)
      std::cout << freq << std::endl;
}