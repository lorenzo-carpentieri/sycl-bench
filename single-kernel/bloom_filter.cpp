#include <iostream>
#include <sycl/sycl.hpp>

#include "common.h"
#include "ParseArgs.h"
#include "ParseData.h"
#include "RandomGenerator.h"

 
namespace s = sycl;

unsigned long djb2Hash(unsigned char* str,int start){
    unsigned long hash = 5381;
    int c;
    while(str[start]!=','){
      c = (int)str[start];
      hash = ((hash<<5)+hash)+c;
      start++;
    }	
    return hash;
  }

unsigned long sdbmHash(unsigned char* str,int start){
  unsigned long hash = 0;
  int c = 0;
  while(str[start]!=','){
    c = (int)str[start];
    hash = c+(hash<<6)+(hash<<16)-hash;
    start++;
  }
  return hash;
}

int calculateIndex(size_t x, size_t y, s::id<2> local_work_item_id, char* dev_words, int wordStartingPosition){
    unsigned long firstValue = djb2Hash((unsigned char*)dev_words,wordStartingPosition) % x;	
    unsigned long secondValue = sdbmHash((unsigned char*)dev_words,wordStartingPosition) % x;
    int fy = local_work_item_id[0];
    if(fy>=y)
      return -1;
    secondValue = (secondValue*fy*fy) % x;
    return (firstValue+secondValue) % x;
}

class QueryWordsKernel{
  private:
    sycl::accessor<char, 1, sycl::access_mode::read>  bloom_filter_acc;
    sycl::accessor<char, 1, sycl::access_mode::read>  words_acc;
    sycl::accessor<int, 1, sycl::access_mode::read>   pos_acc;
    sycl::accessor<char, 1, sycl::access_mode::write> results_acc;
    s::local_accessor<char,1> wordCache;
    int numWords;
    int numHashes;
    int bloomSize;


  public:



    QueryWordsKernel(
      sycl::accessor<char, 1, sycl::access_mode::read> bloom_filter_acc,
      sycl::accessor<char, 1, sycl::access_mode::read> words_acc,   
      sycl::accessor<int, 1, sycl::access_mode::read>  pos_acc,
      sycl::accessor<char, 1, sycl::access_mode::write> results_acc,
      s::local_accessor<char,1> wordCache,
      int numWords,
      int numHashes,
      int bloomSize
      ):
      bloom_filter_acc(bloom_filter_acc),
      words_acc(words_acc),
      pos_acc(pos_acc),
      results_acc(results_acc),
      wordCache(wordCache),
      numWords(numWords),
      numHashes(numHashes),
      bloomSize(bloomSize)
      {};

    void operator()(sycl::nd_item<2>it)const {
      // x-> block[0], y -> block[1]
      auto group = it.get_group();
      s::range<2> block = it.get_local_range();
      s::id<2> block_position = group.get_group_id();
      s::id<2> local_work_item_id = it.get_local_id();
      // Compute on which word apply the hash: this value is used 
      // as index of the position array to understand where is the 
      // first character of the 'current word'
      int currentWord = block[1] * block_position[1] + local_work_item_id[1];

      if(currentWord >= numWords)
        return;
      int blockStartWordIndex = block[1] * block_position[1];
      // position of the first character of the first word in the thread block
      int minPos = pos_acc[blockStartWordIndex];	
      int currentIdx = pos_acc[currentWord];
      if(local_work_item_id[0] == 0){
        int x = 0;
        char currentByte = words_acc[currentIdx];
        for(;currentByte != ','; x++){
          currentByte = words_acc[currentIdx + x];
          wordCache[x + currentIdx - minPos] =  currentByte;	
        }
      }
      s::group_barrier(group);
      int getIdx = calculateIndex(bloomSize, numHashes, local_work_item_id, wordCache.get_pointer(), currentIdx-minPos);
      if(bloom_filter_acc[getIdx]==0){
		    results_acc[currentWord]=0;
	    }
    }
};

class BloomFilterKernel; // kernel forward declaration
   


/*
  A Sobel filter with a convolution matrix 3x3.
  Input and output are two-dimensional buffers of floats.
 */
class BloomFilter {
protected:
  BenchmarkArgs& args;
  size_t bloom_filter_size;
  size_t num_hashes;
  size_t num_bytes_true_file;
  size_t num_bytes_false_file;

  char *input_words;
  char *query_words;
  std::vector<char> bloom_filter;
  size_t num_input_words;
  size_t num_query_words;
  std::vector<int> input_pos;
  std::vector<int> query_pos;
  std::vector<char> query_results;
  std::vector<char> input_results;




  PrefetchedBuffer<char, 1> input_words_buf;
  PrefetchedBuffer<char, 1> query_words_buf;
  PrefetchedBuffer<char, 1> bloom_filter_buf;
  PrefetchedBuffer<int, 1> input_pos_buf;
  PrefetchedBuffer<int, 1> query_pos_buf;
  PrefetchedBuffer<char, 1> query_results_buf;
  PrefetchedBuffer<char, 1> input_results_buf;


  // PrefetchedBuffer<sycl::float4, 2> output_buf;
private:
  s::range<2> calculateBlockDim(int numWords){
    s::device device = args.device_queue.get_device();
    if(numWords == 0 || num_hashes == 0){
      std::cout<< "Specify at least one input word and an hash function" << std::endl;
      return s::range<2>(0,0);
    }
      
    //Firstly, solve for the max number of words that 
    //Can be processed in one thread block.
    size_t maxWorkItemPerGroup = device.get_info<sycl::info::device::max_work_group_size>();
    int maxWordPerGroup = maxWorkItemPerGroup/num_hashes;

    //Check to see if the user demanded more hash functions than
    //A single block can support. If so, only one word per block
    //Will be processed.	
    if(maxWordPerGroup ==0){
      std::cout<< "Error: the work group cannot support this number of hash functions " << std::endl;
      return s::range<2>{0,0};
    }
    //Try to group the words into sets of 32.
    // Be sure that the number of words in a block are multiple of 32
    int wordsPerBlock = 32*(maxWordPerGroup/32);
    if(wordsPerBlock ==0)
      wordsPerBlock = maxWordPerGroup;
    //If all the words can fit in one block.
    if(numWords <= maxWordPerGroup)
      wordsPerBlock = numWords;	
    s::range<2> block(num_hashes,wordsPerBlock);
    return block;
  }

 
public:


  BloomFilter(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    bloom_filter_size = args.problem_size;
    num_hashes = args.cli.getOrDefault<size_t>("--num-hashes", 10);
    num_bytes_true_file = args.cli.getOrDefault<size_t>("--true-file-size", 2500);
    num_bytes_false_file = args.cli.getOrDefault<size_t>("--false-file-size", 2500);

    input_words = (char *) malloc(sizeof(char)*num_bytes_true_file);
    query_words = (char *) malloc(sizeof(char)*num_bytes_false_file);
    bloom_filter.resize(bloom_filter_size);
    std::fill(bloom_filter.begin(), bloom_filter.end(), 0);

      // memset(bloom_filter,0,bloom_filter_size);
  

    num_input_words = generateRandomStrings(input_words, num_bytes_true_file);
    num_query_words = generateRandomStrings(query_words, num_bytes_false_file);
   
    input_pos = getPositions(input_words, num_input_words);
    query_pos = getPositions(query_words, num_query_words);
    
    query_results.resize(num_query_words);
    input_results.resize(num_input_words);

    std::fill(query_results.begin(), query_results.end(), 1);
    std::fill(input_results.begin(), input_results.end(), 1);


    bloom_filter_buf.initialize(args.device_queue, bloom_filter.data(), s::range<1>(bloom_filter_size));
    input_words_buf.initialize(args.device_queue, input_words, s::range<1>(num_bytes_true_file));
    query_words_buf.initialize(args.device_queue, query_words, s::range<1>(num_bytes_false_file));
    input_pos_buf.initialize(args.device_queue, input_pos.data(), s::range<1>(num_input_words));
    query_pos_buf.initialize(args.device_queue, query_pos.data(), s::range<1>(num_query_words));
    query_results_buf.initialize(args.device_queue, query_results.data(), s::range<1>(num_query_words));
    input_results_buf.initialize(args.device_queue, input_results.data(), s::range<1>(num_input_words));

  }

  void run(std::vector<sycl::event>& events) {
    // queue
    sycl::queue q = args.device_queue;
    // device
    sycl::device device = q.get_device();
    
  
    
    events.push_back(q.submit([&](sycl::handler& cgh) {
      // max local memory size in bytes
      size_t max_local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
      if(num_bytes_false_file > max_local_mem_size || num_bytes_true_file > max_local_mem_size){
        std::cout<<"Error the max local size in bytes is: " << max_local_mem_size << std::endl;
        return;
      }

      s::range<2> block{calculateBlockDim(num_input_words)};
      size_t grid_y = num_input_words;
      if(num_input_words % block[1] > 0)
        grid_y += block[1];

      s::range<2> grid{num_hashes, grid_y};

      auto bloom_filter_acc = bloom_filter_buf.get_access<s::access::mode::write>(cgh);
      auto input_words_acc = input_words_buf.get_access<s::access_mode::read>(cgh);
      auto input_pos_acc = input_pos_buf.get_access<s::access_mode::read>(cgh);
      s::local_accessor<char,1> wordCache(s::range<1>(block[1]*51), cgh);

      cgh.parallel_for<class InsertWord>(s::nd_range<2>(grid, block), [=, _bloom_filter_size=bloom_filter_size, _numWords=num_input_words, _numHashes=num_hashes](s::nd_item<2> it){
        // x-> block[0], y -> block[1]
        auto group = it.get_group();
        s::range<2> block = it.get_local_range();
        s::id<2> block_position = group.get_group_id();
        s::id<2> local_work_item_id = it.get_local_id();
        // Compute on which word apply the hash: this value is used 
        // as index of the position array to understand where is the 
        // first character of the 'current word'
        int currentWord = block[1] * block_position[1] + local_work_item_id[1];

        if(currentWord >= _numWords)
              return;
        int blockStartWordIndex = block[1] * block_position[1];
        // position of the first character of the first word in the thread block
        int minPos = input_pos_acc[blockStartWordIndex];	
        int currentIdx = input_pos_acc[currentWord];
        if(local_work_item_id[0] == 0){
          int x = 0;
          char currentByte = input_words_acc[currentIdx];
          for(;currentByte != ','; x++){
            currentByte = input_words_acc[currentIdx + x];
            wordCache[x + currentIdx - minPos] =  currentByte;	
          }
        }
        s::group_barrier(group);
        int setIdx = calculateIndex(_bloom_filter_size, _numHashes, local_work_item_id, wordCache.get_pointer(), currentIdx-minPos);
        if(setIdx<0)
          return;
        bloom_filter_acc[setIdx]=1;
      });

      // parallel for    
    }));

   
    
    events.push_back(q.submit([&](sycl::handler& cgh) {
      // max local memory size in bytes
      size_t max_local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
      if(num_bytes_false_file > max_local_mem_size || num_bytes_true_file > max_local_mem_size){
        std::cout<<"Error the max local size in bytes is: " << max_local_mem_size << std::endl;
        return;
      }

      s::range<2> block{calculateBlockDim(num_input_words)};
      size_t grid_y = (num_input_words/block[1]) * block[1];
      if(num_input_words % block[1] > 0)
        grid_y += block[1];

      s::range<2> grid{num_hashes, grid_y};

      auto bloom_filter_acc = bloom_filter_buf.get_access<s::access::mode::read>(cgh);
      auto input_words_acc = input_words_buf.get_access<s::access_mode::read>(cgh);
      auto input_pos_acc = input_pos_buf.get_access<s::access_mode::read>(cgh);
      auto results_acc = input_results_buf.get_access<s::access_mode::write>(cgh);
      s::local_accessor<char,1> wordCache(s::range<1>(block[1]*51), cgh);
      cgh.parallel_for(s::nd_range<2>(grid, block), 
        QueryWordsKernel(bloom_filter_acc, input_words_acc, input_pos_acc, results_acc, wordCache, num_input_words, num_hashes, bloom_filter_size));
      // parallel for    
    }));   

    events.push_back(q.submit([&](sycl::handler& cgh) {
      // max local memory size in bytes
      size_t max_local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
      if(num_bytes_false_file > max_local_mem_size || num_bytes_true_file > max_local_mem_size){
        std::cout<<"Error the max local size in bytes is: " << max_local_mem_size << std::endl;
        return;
      }

      s::range<2> block{calculateBlockDim(num_query_words)};
      size_t grid_y = (num_query_words / block[1]) * block[1];
      if(num_query_words % block[1] > 0)
        grid_y += block[1];

      s::range<2> grid{num_hashes, grid_y};
      
      auto bloom_filter_acc = bloom_filter_buf.get_access<s::access::mode::read>(cgh);
      auto input_words_acc = query_words_buf.get_access<s::access_mode::read>(cgh);
      auto input_pos_acc = query_pos_buf.get_access<s::access_mode::read>(cgh);
      auto results_acc = query_results_buf.get_access<s::access_mode::write>(cgh);
      s::local_accessor<char,1> wordCache(s::range<1>(block[1]*51), cgh);
      cgh.parallel_for(s::nd_range<2>(grid, block), 
        QueryWordsKernel(bloom_filter_acc, input_words_acc, input_pos_acc, results_acc, wordCache, num_input_words, num_hashes, bloom_filter_size));
      // parallel for    
    }));   
  
  }


  bool verify(VerificationSetting& ver) {
    s::host_accessor<char> bloom_filter_host = bloom_filter_buf.get_host_access();
    s::host_accessor<char> results_input_host = input_results_buf.get_host_access();
    s::host_accessor<char> results_query_host = query_results_buf.get_host_access();

    
    // bloom_filter_buf.reset();
    // for(int i = 0; i < bloom_filter_size; i++){
    //   printf("%d\n", bloom_filter_host[i]);
      // std::cout << bloom_filter[i] << std::endl;
    // }

    for(int i = 0; i < num_input_words; i++){
      if(results_input_host[i]!=1)
        return false;
    }
    
    int num_false=0;
    int num_true=0;
    for(int i = 0; i < num_query_words; i++){
      if(results_query_host[i]==0)
        num_false++;
      else
        num_true++;
    }
    printf("num_true: %d, num_false: %d \n", num_true, num_false);

    return true;
  }


  static std::string getBenchmarkName() { return "BloomFilter"; }

}; // BloomFilter class


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<BloomFilter>();
  return 0;
}