#include <iostream>
#include <sycl/sycl.hpp>

#include "common.h"


#define DELTA_TIME 0.2f
#define EPS2  1e-9f

using namespace sycl;

size_t SIZE_BODY;
size_t BLOCK_SIZE;
int NUM_TILES;

sycl::float3 bodyBodyInteraction(sycl::float4 bi, sycl::float4 bj, sycl::float3 ai)
{
    sycl::float4 r;
    // r_ij [3 FLOPS]
    r = bj - bi;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + EPS2;

    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);
    
    // s = m_j * invDistCube [1 FLOP]
    float s = bj[3] * invDistCube;
    
    // a_i = a_i + s * r_ij [6 FLOPS]
    sycl::float3 tmp  {r[0]*s, r[1]*s, r[2]*s};
    ai =  ai + tmp;
    
    return ai;
}

sycl::float3 tile_calculation(sycl::float4 myPosition, sycl::float3 accel, local_accessor<sycl::float4, 1> sh_position)
{

    for (int i = 0; i < BLOCK_SIZE; i++) {
        accel = bodyBodyInteraction(myPosition, sh_position[i], accel);
    }
    return accel;
}



class calculate_forces{
    private:
        const sycl::accessor<sycl::float4, 1, access_mode::read> in_pos;
        const sycl::accessor<sycl::float4, 1, access_mode::read> in_vel;
        sycl::accessor<sycl::float4, 1, access_mode::read_write> out_pos;
        sycl::accessor<sycl::float4, 1, access_mode::read_write> out_vel;
        sycl::local_accessor<sycl::float4, 1> sh_position;
    
    public:
        calculate_forces(
            const accessor<sycl::float4, 1, access_mode::read> in_pos,
            const accessor<sycl::float4, 1,access_mode::read> in_vel,
            accessor<sycl::float4, 1, access_mode::read_write> out_pos,
            accessor<sycl::float4, 1, access_mode::read_write> out_vel,
            sycl::local_accessor<sycl::float4, 1> sh_position
        )
        :
            in_pos(in_pos),
            in_vel(in_vel),
            out_pos(out_pos),
            out_vel(out_vel),
            sh_position(sh_position){}

        void operator()(sycl::nd_item<1> it) const{
            const auto &group = it.get_group();
            int gtid = it.get_global_id().get(0);
            int local_id = it.get_local_id().get(0);

            sycl::float4 myPosition;
            sycl::float3 acc = {0.0f, 0.0f, 0.0f};
        
            myPosition = in_pos[gtid];

    

            for (int i = 0, tile = 0; i < NUM_TILES; i++, tile++) {
                int idx = tile * BLOCK_SIZE + local_id;
                sh_position[local_id] = in_pos[idx];
                sycl::group_barrier(group);

                acc = tile_calculation(myPosition, acc, sh_position);

                sycl::group_barrier(group);
            }
            // Save the result in global memory for the integration step.
            sycl::float4 acc4 = {acc[0], acc[1], acc[2], 0.0f};

            sycl::float4 oldVel;
            oldVel = in_vel[gtid];
            // updated position and velocity
            sycl::float4 newPos = myPosition + oldVel * DELTA_TIME + acc4 * 0.5f * DELTA_TIME * DELTA_TIME;
            newPos[3] = myPosition[3];
            sycl::float4 newVel = oldVel + (acc4 * DELTA_TIME);
            // write to global memory
            out_pos[gtid] = newPos;
            out_vel[gtid] = newVel;
        }
};







namespace s = sycl;
class NbodyBenchKernel; // kernel forward declaration

/*
  A Sobel filter with a convolution matrix 3x3.
  Input and output are two-dimensional buffers of floats.
 */
class NbodyBench {
protected:
  size_t num_iters;
  std::vector<sycl::float4> pos;
  std::vector<sycl::float4> vel;
  std::vector<sycl::float4> out_pos;
  std::vector<sycl::float4> out_vel;

  size_t w, h; // size of the input picture
  size_t size; // user-defined size (input and output will be size x size)
  size_t local_size;
  BenchmarkArgs& args;


  PrefetchedBuffer<sycl::float4, 1> pos_buff;
  PrefetchedBuffer<sycl::float4, 1> vel_buff;
  PrefetchedBuffer<sycl::float4, 1> out_vel_buff;
  PrefetchedBuffer<sycl::float4, 1> out_pos_buff;

public:
  NbodyBench(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    local_size = args.local_size;
    SIZE_BODY = size;
    BLOCK_SIZE = local_size;
    NUM_TILES = (SIZE_BODY + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iters = args.num_iterations;
    // size is SIZE_BODY
    pos.resize(SIZE_BODY);
    vel.resize(SIZE_BODY);
    out_pos.resize(SIZE_BODY);
    out_vel.resize(SIZE_BODY);

    // Initialization bodies pos and vel
    srand(10);
    for(int i = 0; i < SIZE_BODY; i++){
        // pos
        pos[i][0]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i][1]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i][2]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        // mass
        pos[i][3]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        //vel
        vel[i][0]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i][1]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i][2]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i][3]= 0;
    }
    pos_buff.initialize(args.device_queue, pos.data(), s::range<1>(SIZE_BODY));
    vel_buff.initialize(args.device_queue, vel.data(), s::range<1>(SIZE_BODY));
    out_pos_buff.initialize(args.device_queue, pos.data(), s::range<1>(SIZE_BODY));
    out_vel_buff.initialize(args.device_queue, vel.data(), s::range<1>(SIZE_BODY));
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
        const s::accessor in_pos = pos_buff.get_access<sycl::access::mode::read>(cgh);
        const s::accessor in_vel = vel_buff.get_access<sycl::access::mode::read>(cgh);
        s::accessor out_pos = out_pos_buff.get_access<sycl::access::mode::read_write>(cgh);
        s::accessor out_vel = out_vel_buff.get_access<sycl::access::mode::read_write>(cgh);
        s::local_accessor<sycl::float4,1> sh_position{sycl::range<1>{BLOCK_SIZE},cgh};

        sycl::range<1> block{BLOCK_SIZE};
        sycl::range<1> grid{SIZE_BODY};
        sycl::nd_range<1> ndrange{grid, block};

        cgh.parallel_for<class NbodyBenchKernel>(
            ndrange, calculate_forces(
                in_pos,
                in_vel,
                out_pos,
                out_vel,
                sh_position
            ));
    }));
  }


  bool verify(VerificationSetting& ver) {
    // Triggers writeback

    return true;
  }


  static std::string getBenchmarkName() { return "BoxBlur"; }

}; // NbodyBench class


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<NbodyBench>();
  return 0;
}
