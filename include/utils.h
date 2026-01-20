
#ifndef UTILS_H
#define UTILS_H


#include <array>
#include <type_traits.h>

template <std::size_t... Idx, typename F>
void loop_impl(std::integer_sequence<std::size_t, Idx...>, F&& f) {
  (f(std::integral_constant<std::size_t, Idx>{}), ...);
}

template <std::size_t count, typename F>
void loop(F&& f) {
  loop_impl(std::make_index_sequence<count>{}, std::forward<F>(f));
}

#ifdef __ENABLED_SYNERGY
void polling_freq(synergy::queue q, int to_set, int polling_time_us) {
  int clock_mhz = 0;

  do {
    q.get_synergy_device().set_core_frequency(to_set);
    std::this_thread::sleep_for(std::chrono::microseconds(polling_time_us));
    clock_mhz = q.get_synergy_device().get_core_frequency(false);
    // std::cout<<"Current freq: " << clock_mhz << " / Target freq: " << to_set <<std::endl;
    if (clock_mhz >= to_set - 50 && clock_mhz <= to_set + 50) {
      return ;
    }

    std::cout <<"Current freq: " << clock_mhz << " / Target freq: " << to_set <<std::endl;
  } while (clock_mhz != to_set);
  std::ostringstream poll_info;

  // logs::log_device(poll_info.str());
}
#endif

#endif // UTILS_H

