#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/**
* Prints the help about the program.
* 
*/
void printHelp();

/**
* Determines if a particular argument was specified.
* @param argument The argument being looked for.
* @param args The arguments the user specified.
* @param argc The number of arguments the user specified.
*/
int wasArgSpecified(const char* argument,char** args,int argc);

/**
* Returns the value of the argument..
* @param argument The argument being searched for.
* @param args The argument list specified by the user.
* @param argc The number of arguments specifeid.
* @return Returns the specified value of the argument if it exists.
*/
char* getArgValue(const char* argument,char** args,int argc);

/**
* Responsible for holding information about the bloom filter.
* 
*/
typedef struct BloomOptions{
	/**
	* Holds the number of bits being used in the bloom filter.
	*/
	int size;

	/**
	* Holds the number of hash functions that should be specified.
	*/
	int numHashes;

	/**
	* Holds the size of the batch to be inserted into the filter.
	*/
	int batchSize;

	/**
	* Holds the number of batches to be inserted into the filter.
	*/
	int numBatches;

	
	/**
	* Holds the number of true batches that should be used when querying.
	*/
	int trueBatches;

	/**
	* Holds the number of false batches that should be used when querying.
	*/
	int falseBatches;

	/**
	* Determines the device being used. (Only for cuda)
	*/
	int device;

	/**
	* Determines the filename where the bloom filter should be outputted to.
	*/
	char* fileName;



	

} BloomOptions_t;

/**
* Sets the default parameters of the bloom filter (if a parameter is not specified this value will be used
* @param bloomOptions A pointer to the bloom filter being used.
*/
void setDefault(BloomOptions_t* bloomOptions);

/**
* Parse the bloom filter options specified by the user.
* @param bloomOptions The options to be created.
* @param args The arguments specified by the user.
* @param argc The number of arguments specified by the user. 
*/
void getConfiguration(BloomOptions_t* bloomOptions,char** args,int argc);

/**
* Show the detail of the bloom filter.
*/ 
void showDetails(BloomOptions_t* bloomOptions);

/**
* Responsible for writing the bloom filter to a file.
* @param *bloomOptions A pointer to the options used to describe the bloom filter.
* @param *bloom A pointer to the bloom filter. 
*/
void writeBloomFilterToFile(BloomOptions_t* bloomOptions,char* bloom);


/**
* Prints the help about the program.
* 
*/
void printHelp(){
	printf("\n===============\n");
	printf("--size [num] -s [num] The size of the bloom filter in bits. \n");
	printf("--hashes [num] -h [num] The number of hash functions per word inserted \n");
	printf("--batchSize [num] -b [num] Holds the size of the batch being inserted. \n");
	printf("--help\n");
	printf("--numBatches [num] -n [num] How many batches should be inserted.\n Note, a subset of these will be included as a true batch in the query. \n");
	printf("--generate  Describes if data files should be generated. \n");
	printf("--file [fileName] -f[fileName] Where the bloom filter should be outputted to. \n");
	printf("--trueBatches [num] -tb [num] The size of the subset of the number of	inserted batches that will be queried as true \n");
	printf("--falseBatches [num] -fb [num] Number of false batches \n");
	printf("--numTrueBatchInsertions [num] -ntbi Number of times to insert a true batch. \n This is only for the PBF. \n");
	printf("--prob [float] -p [float] The probabiltiy  (PBF Only) \n");
	printf("--pbfOutput [fileName] The output of the PBF stats. (PBF)\n");
	printf("\n===============\n");
}

/**
* Determines if a particular argument was specified.
* @param argument The argument being looked for.
* @param args The arguments the user specified.
* @param argc The number of arguments the user specified.
*/
int wasArgSpecified(const char* argument,char** args,int argc){
	int i = 0;
	for(i = 0; args[i]!=0; i++){
		if(strcmp(argument,args[i])==0){
			return 1;
		}
	}
	return 0;
}

/**
* Returns the value of the argument..
* @param argument The argument being searched for.
* @param args The argument list specified by the user.
* @param argc The number of arguments specifeid.
* @return Returns the specified value of the argument if it exists.
*/
char* getArgValue(const char* argument,char** args,int argc){
	int i = 0;
	for(i = 0; args[i]!=0 ; i++){
		if(strcmp(argument,args[i])==0){
			if(i<argc-1){
				return args[i+1];
			}
		}
	}
	return 0;
}

/**
* Parse the bloom filter options specified by the user.
* @param bloomOptions The options to be created.
* @param args The arguments specified by the user.
* @param argc The number of arguments specified by the user. 
*/
void getConfiguration(BloomOptions_t* bloomOptions,char** args,int argc){
	char* value = getArgValue("--size",args,argc);
	if(value!=0){
		bloomOptions->size = atoi(value);		
	}
	value = getArgValue("-s",args,argc);
	if(value!=0){
		bloomOptions->size = atoi(value);
	}
	value = getArgValue("--hashes",args,argc); 
	if(value!=0){
		bloomOptions->numHashes = atoi(value);	
	}
	value = getArgValue("-h",args,argc); 
	if(value!=0){
		bloomOptions->numHashes = atoi(value);
	}
	value = getArgValue("--batchSize",args,argc);
	if(value!=0){
		printf("--batchSize");
		bloomOptions->batchSize = atoi(value);	
	}
	value = getArgValue("-b",args,argc);
	if(value!=0){
		bloomOptions->batchSize = atoi(value);
	}
	value = getArgValue("--numBatches",args,argc);
	if(value!=0){
		bloomOptions->numBatches= atoi(value);	
	}
	value = getArgValue("-n",args,argc);
	if(value!=0){
		bloomOptions->numBatches = atoi(value);
	}
	value = getArgValue("--file",args,argc);
	if(value!=0){
		bloomOptions->fileName = value;
	}
	value = getArgValue("-f",args,argc);
	if(value!=0){
		bloomOptions->fileName = value;
	}
	value = getArgValue("--trueBatches",args,argc); 
	if(value!=0){
		bloomOptions->trueBatches = atoi(value);	
	}
	value = getArgValue("-tb",args,argc); 
	if(value!=0){
		bloomOptions->trueBatches = atoi(value);
	}
	value = getArgValue("--falseBatches",args,argc); 
	if(value!=0){
		bloomOptions->falseBatches = atoi(value);	
	}
	value = getArgValue("-fb",args,argc); 
	if(value!=0){
		bloomOptions->falseBatches = atoi(value);
	}

}

/**
* Sets the default parameters of the bloom filter (if a parameter is not specified this value will be used
* @param bloomOptions A pointer to the bloom filter being used.
*/
void setDefault(BloomOptions_t* bloomOptions){
	bloomOptions->size = 100000;	
	bloomOptions->numBatches = 10;
	bloomOptions->numHashes = 10;
	bloomOptions->batchSize = 500;
	bloomOptions->device = 0;
	bloomOptions->fileName = 0;
	bloomOptions->trueBatches=5;
	bloomOptions->falseBatches=5;
}


/**
* Show the detail of the bloom filter.
*/ 
void showDetails(BloomOptions_t* bloomOptions){
	printf("Bloom filter parameters\n");
	printf("size: %i \n",bloomOptions->size);
	printf("numHashes: %i \n",bloomOptions->numHashes);
	printf("batchSize: %i \n",bloomOptions->batchSize);
	printf("numBatches: %i \n",bloomOptions->numBatches);
	printf("numTrueBatches: %i \n",bloomOptions->trueBatches);
	printf("numFalseBatches: %i \n",bloomOptions->falseBatches);
	
}

/**
* Responsible for writing the bloom filter to a file.
* @param *bloomOptions A pointer to the options used to describe the bloom filter.
* @param *bloom A pointer to the bloom filter. 
*/
void writeBloomFilterToFile(BloomOptions_t* bloomOptions,char* bloom){
	//Copy the characters, not the values.
	char* outputBloom = (char*)malloc(sizeof(char)*bloomOptions->size);
	int i = 0;
	for(i = 0; i<bloomOptions->size;i++){
		outputBloom[i] = (bloom[i])? '1':'0';
	}
	FILE* bloomFile = fopen(bloomOptions->fileName,"w+");
	fwrite(outputBloom,sizeof(char),bloomOptions->size,bloomFile);
	fclose(bloomFile);
	free(outputBloom);
}

