#include <synergy.hpp>

#define N 100000
template<typename Bench>
class A{
    private:
        Bench b;
    public:
        A(){};
        A(Bench b):
        b(b){}

        void run(){
            std::vector<sycl::event> run_events;
            run_events.reserve(1024);
            synergy::queue q{};
            b.run(run_events, q);
            for(sycl::event& e :run_events){
                q.kernel_energy_consumption(e);
            }

        }
        

};

class MyBench{
    public:
        MyBench(){};
        void run(std::vector<sycl::event>& events,synergy::queue& q ) {
            std::vector<int> a(N);
            std::fill(a.begin(), a.end(), 1);
            
            sycl::buffer<int, 1> buf_a{a.data(), N};
            
            
            sycl::event e = q.submit([&](sycl::handler& cgh){
                    sycl::accessor<int, 1, sycl::access_mode::read_write> acc_a {buf_a, cgh};
                    cgh.parallel_for(
                        sycl::range<1>{N}, [=](sycl::id<1> id){ 
                            acc_a[id] = acc_a[id]*2+acc_a[id];
                        }
                    );
                }
                ); // end submit
            events.push_back(e);
            std::cout<< q.kernel_energy_consumption(e) <<std::endl;
            sycl::host_accessor<int,1> host_acc = buf_a.get_host_access();
            std::cout<< host_acc[0]<< std::endl;

        }
};
int main(){
    
    A<MyBench>(MyBench{}).run();
}