#ifndef SYCL_UTIL_FUNCTS_H
#define SYCL_UTIL_FUNCTS_H

#include <sycl/sycl.hpp>
#include "../../include/queue_macro.h"

template <typename T, int Dims>
void initDeviceBuffer(selected_queue& queue, sycl::buffer<T, Dims>& buffer, T* data) {
	using namespace sycl;

	queue.submit([&](handler& cgh) {
		auto accessor = buffer.template get_access<access::mode::discard_write>(cgh);
		cgh.copy(data, accessor);
	});

	queue.wait();
}

#endif
