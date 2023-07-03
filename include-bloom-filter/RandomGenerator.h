#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
* Responsible for creating random data of specified sizes.
* @param numBytes The length of the String in bytes.
*/
char* generateRandomString(int numBytes){
	char* randomString = (char*)malloc(numBytes);
	int i =0;
	for(i =0; i<numBytes-1;i++){
		randomString[i] = random()%57+65;
	}
	randomString[numBytes-1]=',';
	return randomString;
}

int generateRandomStrings(char *input_words, int numBytes){
	int totalBytesAdded = 0;
	int numWords = 0;
	// input_words = (char*)malloc(sizeof(char)*numBytes); 
	while(1){
		int numBytesString = random()%50+2;
		char* randomString = generateRandomString(numBytesString);
		if(totalBytesAdded+numBytesString>=numBytes){
			free(randomString);
			break;
		}
		memcpy(input_words+totalBytesAdded,randomString,numBytesString);
		free(randomString);
		totalBytesAdded+=numBytesString;
		numWords++;
	}

	return numWords;
}
