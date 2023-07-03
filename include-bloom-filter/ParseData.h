#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
/**
* Holds information about the words read from the file, including the number of words, a char
* pointer to the array of strings read, and an array of integers corresponding
* to the starting and stopping positions of each word in the array. In positions, each even 
* integer corresponds to the starting point of the word, and each odd number corresponds to
* the ending point of a word.
*/
typedef struct WordAttributes{
	char* currentWords;
	int* positions;
	int numWords;
	int numBytes;
} WordAttributes;

/**
* Holds a list of positions for each word.
*/
typedef struct Positions{
	int currentStartPosition;
	int currentEndPosition;
	struct Positions* nextPosition;	
} Positions;

/**
* Turns a double array of character pointers into a single array
* of characters separated by commas.
* @param words The words being turned into CSV format.
* @param numWords The number of words in total.
* @return Returns the words in a single, csv array.
*/
char* toCSV(char** words,int numWords);

/**
* Parses the WordAttributes of a file.
* @param fileData The data being parsed.
* @param fileSize The number of bytes in the data.
* @return The WordAttributes that describes the data.
*/
WordAttributes* parseWordAttributes(char* fileData,int fileSize);

/**
* Loads files
* @param index The index of the file being loaded.
* @return Returns a WordAttributes structure desribing the words that were parsed. 
*/
WordAttributes* loadFile(int index);

/**
* Loads files by prefix
* @param index The index of the file being loaded.
* @param prefix The prefix to the filename.
* @return Returns a WordAttributes structure desribing the words that were parsed. 
*/
WordAttributes* loadFileByPrefix(int index,char* prefix);


/**
* Loads files specified by name.
* @param fileName The name of the file being opened.
* @return Returns a WordAttributes structure describing the words that were parsed.
*/
WordAttributes* loadFileByName(char* newFile);

/**
* Converts the list of Positions to an array of integers and garbage collects the list.
* @firstPosition The first position in the list of positions.
* @numWords The number of words specified.
* @return Returns an array of integers corresponding to the different positions.
*/
int* cleanAndGetPositions(Positions* firstPosition,int numWords);

/**
* Frees the memory allocated by the WordAttributes.
* @param wordAttributes A pointer to the word attributes being freed.
*/
void freeWordAttributes(WordAttributes* wordAttributes);

/**
* Responsible for splitting a WordAttributes into two WordAttributes.
*/
WordAttributes* splitWordAttributes(WordAttributes* original);



/**
* Turns a double array of character pointers into a single array
* of characters separated by commas.
* @param words The words being turned into CSV format.
* @param numWords The number of words in total.
* @return Returns the words in a single, csv array.
*/
char* toCSV(char** words,int numWords){
	if(numWords<=0){
		return 0;
	}
	int firstWordLength = strlen(words[numWords-1]);
	char* singleArray = (char*)malloc(sizeof(char)*firstWordLength+sizeof(char));
	memset(singleArray,0,firstWordLength+1);
	memcpy(singleArray,words[numWords-1],firstWordLength);
	singleArray[firstWordLength] = ',';
	numWords-=2;
	while(numWords>=0){
		int wordLength = strlen(words[numWords]);
		singleArray = (char*)realloc(singleArray,sizeof(char)*wordLength+1);
		strcat(singleArray,words[numWords]);
		strcat(singleArray,",");	 
		numWords--;	
	}	
	return singleArray;
}




std::vector<int> getPositions(char *str, int num_word){
	std::vector<int> positions(num_word);
	// char* newWord = strtok(str,",");
	// int currentStartPosition = 0;
	// int currentEndPosition = strlen(newWord);
	// positions[0] = 0;
	// int i = 1;
	// while (newWord)
	// {	
	// 	positions[i] = currentEndPosition + 1;
	// 	currentStartPosition = positions[i];
	// 	newWord = strtok(NULL,",");
		
		
	// 	// if(newWord == 0)
	// 	// 	break ;
		
	// 	currentEndPosition = currentStartPosition + strlen(newWord);
	// 	// free(newWord);
	// 	i++;
	// }
	
	positions[0]=0;
	for(int i  =  0, j = 1; i < strlen(str) && j < num_word; i++){
		if(str[i]==','){
			positions[j]=i+1;
			j++;
		}
	}
	
	
	return positions;
}


