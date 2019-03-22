/* -----------  FastPlace - Version 3.1 ----------------
                       by 
   Natarajan Viswanathan and Chris C.-N. Chu
     Dept. of ECpE, Iowa State University
          Copyright (c) - 2004 
Iowa State University Research Foundation, Inc.
--------------------------------------------------------*/
/* --------------------------------------------------------------------------
   Contains routines to:
   - Read and Write the benchmark files in Bookshelf format 
----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "memAlloc.h"
#include "bookshelf_IO.h"

#define hashsize(n) ((uint32_t)1<<(n))
#define hashmask(n) (hashsize(n)-1)
#define rot(x, k) (((x)<<(k)) | ((x)>>(32-(k))))
#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}
#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c,4);  \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}


/*-- extern variables --*/
    // from createHash()
    char **cellName = NULL;
    
    // from readAuxFile()
    char *nodesFile = NULL;
    char *netsFile = NULL;
    char *wtsFile = NULL;
    char *sclFile = NULL;
    char *plFile = NULL;
    char *benchmarkName = NULL;
    
    // from readNodesFile()
    int movableNodes, numTerminals;
    float averageCellWidth;
    float *cellWidth = NULL;
    float *cellHeight = NULL;
    float *cellArea = NULL;
    
    // from readNetsFile() 
    int numNets, numPins;
    int *netlist = NULL;
    int *netlistIndex = NULL;
    float *xPinOffset = NULL;
    float *yPinOffset = NULL;
    char **netName = NULL;
    
    // from readPlFile()
    float *xCellCoord = NULL;
    float *yCellCoord = NULL;
    char **cellOrientation = NULL;
    
    // from readSclFile()
    ROW *core_row = NULL;
    int numRows, numRowBlockages;
    float siteOriginY, siteEndY, coreHeight;
    float siteOriginX, siteEndX, coreWidth;
    float minX, maxX, minY, maxY;
    float coreRowHeight;
    float *xRowBlockage = NULL;
    float *yRowBlockage = NULL;
    float *widthRowBlockage = NULL;
    
    // from readWtsFile()
    float *netWeight = NULL;
    
    
/*-- global variables --*/
    typedef struct nodesHash NODES;
    struct nodesHash  {
        char *name;
        unsigned long index;
    };
    NODES **NodesInfo = NULL;
    NODES **NetsInfo = NULL;
    long hashSize, hashBits;
    unsigned long *RN = NULL;
    unsigned char *hashFlag = NULL;
    
    int numNodes;
    char error_text[BUFFERSIZE];


/*-- functions --*/
    void createHash(char benchmarkPath[], char nodesFile[]);
    /*inline*/ uint32_t get_hashfunc(const void *key, size_t length, uint32_t initval);
    void freeHash();
    void readAuxFile(char benchmarkPath[], char auxFile[]);
    /*inline*/ long getIndex(char temp[]);
    void readNodesFile(char benchmarkPath[], char nodesFile[]);
    void readNetsFile(char benchmarkPath[], char netsFile[]);
    void readPlFile(char benchmarkPath[], char plFile[]);
    void readSclFile(char benchmarkPath[], char sclFile[]);
    void readWtsFile(char benchmarkPath[], char wtsFile[], int net_weights);
    void writePlFile(char outputDir[], char fileName[], float xCoord[], float yCoord[], 
                     int totalNodes);


/* -----------------------------------------------------------
   Reads the .nodes file and creates a hash linking cell name 
   to cell index for all the nodes in the circuit 
   (movable nodes + fixed nodes + I/O pads)
   
   creates extern vars:
      cellName[]
----------------------------------------------------------- */
void createHash(char benchmarkPath[], char nodesFile[]) 
{
    FILE *fp;
    char file_name[BUFFERSIZE], temp[BUFFERSIZE], node_type[50];
    char line[LINESIZE];
    float nodeWidth, nodeHeight;
    long j, nonpin_ptr, pin_ptr;
    long R, hashfunc, RN_index;
    int node_count;
    
    
    strcpy(file_name, benchmarkPath);
    strcat(file_name, "/");
    strcat(file_name, nodesFile);
    
    if((fp=fopen(file_name, "r")) == NULL) {
        sprintf(error_text, "bookshelf_IO: Cannot open: %s file", file_name);
        runtimeError(error_text);
    }
    
    numNodes = numTerminals = 0;
    while(!feof(fp)) {
        fgets(line, LINESIZE, fp);
        sscanf(line, "%s\t%*s\n", temp);
        
        if(strlen(line) < 5 || temp[0] == '#' || strcmp(temp, "UCLA") == 0)
            continue;
        
        if(strcmp(temp, "NumNodes") == 0) {
            sscanf(line, "NumNodes\t:\t%d\n", &numNodes);
        } else if(strcmp(temp, "NumTerminals") == 0) {
            sscanf(line, "NumTerminals\t:\t%d\n", &numTerminals);
            break;
        } else {}    
    }
    fclose(fp);
    
    if(numNodes == 0) {
        sprintf(error_text, "bookshelf_IO: NumNodes = %d", numNodes);
        runtimeError(error_text);
    }
    
    // defining and allocating hash variables
    hashBits = 3+(long)(log(numNodes)/log(2));
    hashSize = hashsize(hashBits);
    
    if(NodesInfo == NULL) {
        NodesInfo = (NODES **) malloc(hashSize*sizeof(NODES *));
        if(!NodesInfo)
            runtimeError("bookshelf_IO: allocation failure in createNodesHash()");
    } else {
        runtimeError("bookshelf_IO: NodesHash is not NULL");
    }
    for(j=0; j<hashSize; j++)
        NodesInfo[j] = NULL;
    
    if(RN == NULL && hashFlag == NULL) {
        RN = lvector(1, hashSize);
        hashFlag = cvector(1, hashSize);
    } else {
        runtimeError("bookshelf_IO: error while creating input hash");
    }
    
    // global vector giving inverse mapping b/w cell names and cell indexes
    if(cellName == NULL) {
        cellName = (char **) malloc((numNodes+1)*sizeof(char *));
        if(!cellName)
            runtimeError("bookshelf_IO: allocation failure for cellName");
        for(j=0; j<=numNodes; j++)
            cellName[j] = NULL;
    } else {
        runtimeError("bookshelf_IO: error while creating input hash");
    }
    
    // initialize hash flags
    for(j=1; j<=hashSize; j++)
        hashFlag[j] = 0;
    
    // generate random sequence
    R = 1;
    for(j=1; j<=hashSize; j++) {
        R = ((5*R) & hashmask(hashBits));
        RN[j] = R/4;
    }
    
    if((fp=fopen(file_name, "r")) == NULL) {
        sprintf(error_text, "bookshelf_IO: Cannot open: %s file", file_name);
        runtimeError(error_text);
    }
    
    movableNodes = numNodes - numTerminals;
    nonpin_ptr = 1;                      // movable nodes start from 1
    pin_ptr = numNodes-numTerminals+1;   // fixed nodes start from movableNodes+1
    node_count = 0;
    
    while(!feof(fp)) {
        *line = '\0';
        fgets(line, LINESIZE, fp);
        sscanf(line, "%s\t%*s\n", temp);
        
        // intermediate lines can be blanks or begin with #
        if(strlen(line) < 5 || temp[0] == '#' || strcmp(temp, "UCLA") == 0)
            continue;
        
        if(strcmp(temp, "NumNodes") == 0 || strcmp(temp, "NumTerminals") == 0)
            continue;
        
        *node_type = '\0';
        sscanf(line, "%s%f%f%s\n", temp, &nodeWidth, &nodeHeight, node_type);
        node_count++;
        
        if((strcmp(node_type, "terminal") == 0) || 
           (strcmp(node_type, "terminal_NI") == 0)) {
            
            // create array to save cell name
            cellName[pin_ptr] = (char *) malloc((strlen(temp)+1)*sizeof(char));
            strcpy(cellName[pin_ptr], temp);
            
            // create a hash table for name searching
            hashfunc = (long)get_hashfunc(cellName[pin_ptr], 
                                          strlen(cellName[pin_ptr])*sizeof(char), 
                                          13);
            hashfunc = (hashfunc & hashmask(hashBits));
            
            RN_index = 1;
            while(hashFlag[hashfunc] != 0 && RN_index < hashSize) {
                hashfunc = ((hashfunc + RN[RN_index]) & hashmask(hashBits));
                RN_index++;
            }
            
            if(hashfunc >= hashSize) {
                sprintf(error_text, "bookshelf_IO: Cannot fill %s in hash table", temp);
                runtimeError(error_text);
            }
            
            NodesInfo[hashfunc] = (NODES *) malloc(sizeof(NODES));
            NodesInfo[hashfunc]->name  = cellName[pin_ptr];
            NodesInfo[hashfunc]->index = pin_ptr;
            hashFlag[hashfunc] = 1;
            
            pin_ptr++;
        
        } else {
            
            // create array to save cell name
            cellName[nonpin_ptr] = (char *) malloc((strlen(temp)+1)*sizeof(char));
            strcpy(cellName[nonpin_ptr], temp);
            
            // create a hash table for name searching
            hashfunc = (long)get_hashfunc(cellName[nonpin_ptr], 
                                          strlen(cellName[nonpin_ptr])*sizeof(char), 
                                          13);
            hashfunc = (hashfunc & hashmask(hashBits));
            
            RN_index = 1;
            while(hashFlag[hashfunc] != 0 && RN_index < hashSize) {
                hashfunc = ((hashfunc + RN[RN_index]) & hashmask(hashBits));
                RN_index++;
            }
            
            if(hashfunc >= hashSize) {
                sprintf(error_text, "bookshelf_IO: Cannot fill %s in hash table", temp);
                runtimeError(error_text);
            }
            
            NodesInfo[hashfunc] = (NODES *) malloc(sizeof(NODES));
            NodesInfo[hashfunc]->name  = cellName[nonpin_ptr];
            NodesInfo[hashfunc]->index = nonpin_ptr;
            hashFlag[hashfunc] = 1;
            
            nonpin_ptr++;
        }
    }
    
    fclose(fp);
    if(node_count != numNodes) {
        sprintf(error_text, "bookshelf_IO: NumNodes (%d) != Number of Node descriptions (%d)", 
                numNodes, node_count);
        runtimeError(error_text);
    }
}


/* -----------------------------------------------------------------------
  hash function generator
  
  By Bob Jenkins, 2006.  bob_jenkins@burtleburtle.net.  You may use this
  code any way you wish, private, educational, or commercial.  It's free.
  http://burtleburtle.net/bob/hash/doobs.html
----------------------------------------------------------------------- */
inline uint32_t get_hashfunc(const void *key, size_t length, uint32_t initval)
{
  uint32_t a,b,c;       // internal state
  
  
  // Set up the internal state
  a = b = c = 0xdeadbeef + ((uint32_t)length) + initval;
  
  // need to read the key one byte at a time
  const uint8_t *k = (const uint8_t *)key;
  
  // all but the last block: affect some 32 bits of (a,b,c)
  while (length > 12) {
      a += k[0];
      a += ((uint32_t)k[1])<<8;
      a += ((uint32_t)k[2])<<16;
      a += ((uint32_t)k[3])<<24;
      b += k[4];
      b += ((uint32_t)k[5])<<8;
      b += ((uint32_t)k[6])<<16;
      b += ((uint32_t)k[7])<<24;
      c += k[8];
      c += ((uint32_t)k[9])<<8;
      c += ((uint32_t)k[10])<<16;
      c += ((uint32_t)k[11])<<24;
      mix(a,b,c);
      length -= 12;
      k += 12;
  }
  
  // last block: affect all 32 bits of (c)
  switch(length) {                     // all the case statements fall through
      case 12: c+=((uint32_t)k[11])<<24;
      case 11: c+=((uint32_t)k[10])<<16;
      case 10: c+=((uint32_t)k[9])<<8;
      case 9 : c+=k[8];
      case 8 : b+=((uint32_t)k[7])<<24;
      case 7 : b+=((uint32_t)k[6])<<16;
      case 6 : b+=((uint32_t)k[5])<<8;
      case 5 : b+=k[4];
      case 4 : a+=((uint32_t)k[3])<<24;
      case 3 : a+=((uint32_t)k[2])<<16;
      case 2 : a+=((uint32_t)k[1])<<8;
      case 1 : a+=k[0];
               break;
      case 0 : return c;
  }
  
  final(a,b,c);
  
  return c;
}

    
/* -----------------------------------------------------------
  frees hash elements
----------------------------------------------------------- */
void freeHash()
{
    int j;
    
    
    for(j=0; j<hashSize; j++) {
        if(NodesInfo[j]) {
            free(NodesInfo[j]);
            NodesInfo[j] = NULL;
        }
    }
    free(NodesInfo);
    NodesInfo = NULL;
    
    free_lvector(RN, 1, hashSize);
    RN = NULL;
    
    free_cvector(hashFlag, 1, hashSize);
    hashFlag = NULL;
}


/* -----------------------------------------------------------
  Reads the .aux file to get the other file names
  
  creates extern vars:
     nodesFile[], netsFile[], wtsFile[], sclFile[], 
     plFile[], benchmarkName[];
----------------------------------------------------------- */
void readAuxFile(char benchmarkPath[], char auxFile[]) 
{
    FILE *fp;
    char temp[BUFFERSIZE], placementType[BUFFERSIZE], *name;
    
    
    strcpy(temp, benchmarkPath);
    strcat(temp, "/");
    strcat(temp, auxFile);
    
    nodesFile = cvector(0, BUFFERSIZE);
    netsFile = cvector(0, BUFFERSIZE);
    wtsFile = cvector(0, BUFFERSIZE);
    sclFile = cvector(0, BUFFERSIZE);
    plFile = cvector(0, BUFFERSIZE);
    benchmarkName = cvector(0, BUFFERSIZE);
    
    if((fp=fopen(temp, "r")) == NULL) {
        sprintf(error_text, "bookshelf_IO: Cannot open: %s file", auxFile);
        runtimeError(error_text);
    }
//    printf("Reading %s ...\n",auxFile);
    
    fscanf(fp,"%s\t:\t%s%s%s%s%s\n", placementType, nodesFile, netsFile, wtsFile, plFile, sclFile);
    
    strcpy(temp, auxFile);
    name = strtok(temp, ".");
    strcpy(benchmarkName, name);
    
    fclose(fp);
}  


inline long getIndex(char temp[])
{
    long hashfunc, RN_index;
    
    
    // find the nodeIndex corresponding to temp
    hashfunc = (long)get_hashfunc(temp, strlen(temp)*sizeof(char), 13);
    hashfunc = (hashfunc & hashmask(hashBits));
    
    RN_index = 1;
    while(RN_index < hashSize) {
        if(NodesInfo[hashfunc] && strcmp(temp, NodesInfo[hashfunc]->name) == 0) {
            break;
        } else {
            hashfunc = ((hashfunc + RN[RN_index]) & hashmask(hashBits));
            RN_index++;
        }
    }
    
    if(RN_index >= hashSize) {
        sprintf(error_text, "bookshelf_IO: Cannot find %s in hash table", temp);
        runtimeError(error_text);
    }
    
    return NodesInfo[hashfunc]->index;
}


/* -----------------------------------------------------------
  Reads the .nodes file to get cell widths and heights
  
  creates extern vars: 
     movableNodes, numTerminals, averageCellWidth, 
     cellWidth[], cellHeight[], cellArea[]
----------------------------------------------------------- */
void readNodesFile(char benchmarkPath[], char nodesFile[])
{
    FILE *fp;
    char file_name[BUFFERSIZE], temp[BUFFERSIZE], node_type[50];
    char line[LINESIZE];
    long nodeIndex;
    float nodeWidth, nodeHeight, sumWidth;
    int node_count;
    
    
    strcpy(file_name, benchmarkPath);
    strcat(file_name, "/");
    strcat(file_name, nodesFile);
    
    if((fp=fopen(file_name, "r")) == NULL) {
        sprintf(error_text, "bookshelf_IO: Cannot open: %s file", file_name);
        runtimeError(error_text);
    }
//    printf("Reading %s ...\n", nodesFile);
    
    movableNodes = numNodes - numTerminals;       // global var - num of movable cells
    if(cellWidth == NULL && cellHeight == NULL && cellArea == NULL) {
        cellWidth  = vector(1, numNodes);         // global vector giving cell widths
        cellHeight = vector(1, numNodes);         // global vector giving cell heights
        cellArea   = vector(1, numNodes);         // global vector giving cell areas
    } else {
        runtimeError("bookshelf_IO: allocation error in readNodesFile()");
    }
    
    sumWidth = 0;
    node_count = 0;
    
    while(!feof(fp)) {
        *line = '\0';
        fgets(line, LINESIZE, fp);
        sscanf(line, "%s\t%*s\n", temp);
        
        // intermediate lines can be blanks or begin with #
        if(strlen(line) < 5 || temp[0] == '#' || strcmp(temp, "UCLA") == 0)
            continue;
        
        if(strcmp(temp, "NumNodes") == 0 || strcmp(temp, "NumTerminals") == 0)
            continue;
        
        *node_type = '\0';
        sscanf(line, "%s%f%f%s\n", temp, &nodeWidth, &nodeHeight, node_type);
        node_count++;
        
        nodeIndex = getIndex(temp);
        
        // store cellwidth, cellheight and cellarea corresponding to nodeIndex
        cellWidth[nodeIndex]  = nodeWidth;
        cellHeight[nodeIndex] = nodeHeight;
        cellArea[nodeIndex]   = nodeWidth*nodeHeight;
        if(nodeIndex <= movableNodes)
            sumWidth += nodeWidth;
    }
    
    // find average cell width
    averageCellWidth = sumWidth/movableNodes;
    averageCellWidth *= 100;
    averageCellWidth = (int)averageCellWidth;
    averageCellWidth /= 100;
    
    fclose(fp);
    if(node_count != numNodes) {
        sprintf(error_text, "bookshelf_IO: NumNodes (%d) != Number of Node descriptions (%d)", 
                numNodes, node_count);
        runtimeError(error_text);
    }
    
#if(DEBUG)
int i;
for(i=1; i<=movableNodes+numTerminals; i++) {
    printf("%d  %s  %.2f  %.2f  %.2f\n", i, cellName[i], cellWidth[i], cellHeight[i], cellArea[i]);
}

printf("Avg Cell Width:  %.2f \n", averageCellWidth);    
#endif
}


/* -----------------------------------------------------------
   Reads the .nets file to get the netlist information
   
   creates extern vars: 
      numNets, numPins, 
      xPinOffset[], yPinOffset[], netlist[], netlistIndex[]
----------------------------------------------------------- */
void readNetsFile(char benchmarkPath[], char netsFile[])
{
    FILE *fp;
    char file_name[BUFFERSIZE], temp[BUFFERSIZE], nodeName[BUFFERSIZE], pinDirection[5];
    char line[LINESIZE];
    long nodeIndex;
    int degree, prevElements, netNo, k, startPointer;
    float xOffset, yOffset;
    
    
    strcpy(file_name, benchmarkPath);
    strcat(file_name, "/");
    strcat(file_name, netsFile);
    
    if((fp=fopen(file_name, "r"))==NULL) {
        sprintf(error_text, "bookshelf_IO: Cannot open %s file", file_name);
        runtimeError(error_text);
    }
    
    numNets = numPins = 0;
    while(!feof(fp)) {
        fgets(line, LINESIZE, fp);
        sscanf(line, "%s\t%*s\n", temp);
        
        if(strlen(line) < 5 || temp[0] == '#' || strcmp(temp, "UCLA") == 0)
            continue;
        
        if(strcmp(temp, "NumNets") == 0) {
            sscanf(line, "NumNets\t:\t%d\n", &numNets);
        } else if(strcmp(temp, "NumPins") == 0) {
            sscanf(line, "NumPins\t:\t%d\n", &numPins);
            break;
        } else {}
    }
    fclose(fp);
    
    if(numNets == 0 || numPins == 0) {
        sprintf(error_text, "bookshelf_IO: NumNets = %d,     NumPins = %d ", numNets, numPins);
        runtimeError(error_text);
    }
    
    if(netlist == NULL && xPinOffset == NULL && yPinOffset == NULL &&
       netlistIndex == NULL) {
        // stores the netlist and pin offsets relative to the center of the cells
        netlist    = ivector(1,numPins+1);
        xPinOffset = vector(1,numPins+1);
        yPinOffset = vector(1,numPins+1);
        
        // index vector for the netlist and offset vectors
        netlistIndex = ivector(0,numNets+1);
    } else {
        runtimeError("bookshelf_IO: allocation error in readNetsFile()");
    }
    
    // inverse map b/w net name and net no
    netName = (char **) malloc((numNets+1)*sizeof(char *));
    
    if((fp=fopen(file_name, "r"))==NULL) {
        sprintf(error_text, "bookshelf_IO: Cannot open %s file", file_name);
        runtimeError(error_text);
    }
//    printf("Reading %s ...\n", netsFile);
    
    startPointer = k = 1;
    netlistIndex[0] = 1;
    prevElements = 0;
    netNo = 0;
    
    while(!feof(fp)) {
        *line = '\0';
        fgets(line, LINESIZE, fp);
        sscanf(line, "%s\t%*s\n", temp);
        
        if(strlen(line) < 5 || temp[0] == '#' || strcmp(temp, "UCLA") == 0)
            continue;
        
        if(strcmp(temp, "NumNets") == 0 || strcmp(temp, "NumPins") == 0)
            continue;
        
        if(strcmp(temp, "NetDegree") == 0) {
            netNo++;
            *temp = '\0';
            sscanf(line, "NetDegree\t:\t%d\t%s\n", &degree, temp);
            if(strcmp(temp, "") == 0)
                sprintf(temp, "net_%d", netNo);
            netName[netNo] = (char *) malloc((strlen(temp)+1)*sizeof(char));
            strcpy(netName[netNo], temp);
            
            netlistIndex[netNo] = netlistIndex[netNo-1] + prevElements;
            startPointer = netlistIndex[netNo];
            prevElements = degree;
            k = 1;
        } else {
            xOffset = yOffset = 0.0;
            *pinDirection = '\0';
            
            sscanf(line, "%s%s", nodeName, pinDirection);
            if(pinDirection[0] == ':')
                sscanf(line, "%*s%*s%f%f", &xOffset, &yOffset);
            else
                sscanf(line, "%s%s%*s%f%f", nodeName, pinDirection, &xOffset, &yOffset);
            
            if(strcmp(pinDirection, "") == 0 || pinDirection[0] == ':')
                strcpy(pinDirection, "B");
            
            nodeIndex = getIndex(nodeName);
            
            netlist[startPointer+k-1]    = nodeIndex;
            xPinOffset[startPointer+k-1] = xOffset;
            yPinOffset[startPointer+k-1] = yOffset;
            k++;
        }
    }
    netlistIndex[numNets+1] = netlistIndex[numNets] + prevElements;
    netlist[netlistIndex[numNets+1]] = 0;
    
    fclose(fp); 
    if(netNo != numNets) {
        sprintf(error_text, "bookshelf_IO: NumNets (%d) != Number of Net descriptions (%d)", 
                numNets, netNo);
        runtimeError(error_text);
    }
    
    if(netlistIndex[numNets+1]-1 != numPins) {
        sprintf(error_text, "bookshelf_IO: NumPins (%d) != Number of Pin descriptions (%d)", 
                numPins, netlistIndex[numNets+1]-1);
        runtimeError(error_text);
    }
    
#if(DEBUG)
int i, j;
for(i=1; i<=numNets; i++) {
    printf("**%d**  ", netlistIndex[i+1]-netlistIndex[i]);
    for(j=netlistIndex[i]; j<netlistIndex[i+1]; j++) {
        printf("(%d) %.2f %.2f  ", netlist[j], xPinOffset[j], yPinOffset[j]);
    }
    printf("\n");    
}
#endif
}


/* -----------------------------------------------------------
  Reads the .pl file to get coordinates of all nodes and the 
  placement boundary based on the position of the I/O pads
  
  creates extern vars:
     xCellCoord[], yCellCoord[]
----------------------------------------------------------- */
void readPlFile(char benchmarkPath[], char plFile[])
{
    FILE *fp;
    char temp[BUFFERSIZE], nodeName[BUFFERSIZE], fixedType[50], orientation[5];
    char line[LINESIZE];
    long nodeIndex;
    int node_count;
    float xCoord, yCoord;
    
    
    strcpy(temp, benchmarkPath);
    strcat(temp, "/");
    strcat(temp, plFile);
    
    if((fp=fopen(temp, "r"))==NULL) {
        sprintf(error_text, "bookshelf_IO: Cannot open %s file", temp);
        runtimeError(error_text);
    }
//    printf("Reading %s ...\n", plFile);
    
    if(xCellCoord == NULL && yCellCoord == NULL && cellOrientation == NULL) {
        xCellCoord = vector(1,numNodes);
        yCellCoord = vector(1,numNodes);
        cellOrientation = cmatrix(1, numNodes, 0, 4);
    } else {
        runtimeError("bookshelf_IO: allocation error in readPlFile()");
    }
    
    node_count = 0;
    
    while(!feof(fp)) {
        *line = '\0';
        fgets(line, LINESIZE, fp);
        sscanf(line, "%s\t%*s\n", temp);
        
        if(strlen(line) < 5 || temp[0] == '#' || strcmp(temp, "UCLA") == 0)
            continue;
        
        *fixedType = '\0';
        strcpy(orientation, "N");
        sscanf(line, "%s%f%f\t:\t%s%s\n", nodeName, &xCoord, &yCoord, orientation, fixedType);
        
        node_count++;
        nodeIndex = getIndex(nodeName);
        // Assume all coordinates are integers for FLUTE
        xCellCoord[nodeIndex] = (int) (xCoord + 0.5*cellWidth[nodeIndex]);
        yCellCoord[nodeIndex] = (int) (yCoord + 0.5*cellHeight[nodeIndex]);
        strcpy(cellOrientation[nodeIndex], orientation);
    }
    
    fclose(fp);
    if(numNodes != node_count) {
        sprintf(error_text, "bookshelf_IO: NumNodes (%d) != Number of Node descriptions (%d)", 
                numNodes, node_count);
        runtimeError(error_text);
    }
}


/* -----------------------------------------------------------
  Reads the .scl file to get placement region information
  
  creates extern vars:
     siteOriginX, siteEndX, siteOriginY, siteEndY
     coreRowHeight, coreWidth, coreHeight, numRows 
     minX, maxX, minY, maxY
     core_row[] data structure
     xRowBlockage[], yRowBlockage[], widthRowBlockage[]
----------------------------------------------------------- */
void readSclFile(char benchmarkPath[], char sclFile[])
{
    FILE *fp;
    char file_name[BUFFERSIZE], temp[BUFFERSIZE];
    char line[LINESIZE];
    float subroworigin;
    int j, k, totalSites, row_count, subrow_count, rowBlockageCount;
    
    
    strcpy(file_name, benchmarkPath);
    strcat(file_name, "/");
    strcat(file_name, sclFile);
    
    if((fp=fopen(file_name, "r"))==NULL) {
        sprintf(error_text, "bookshelf_IO: Cannot open %s file", file_name);
        runtimeError(error_text);
    }
    
    numRows = 0;
    while(!feof(fp)) {
        fgets(line, LINESIZE, fp);
        sscanf(line, "%s\t%*s\n", temp);
        
        if(strlen(line) < 5 || temp[0] == '#' || strcmp(temp, "UCLA") == 0)
            continue;
        
        if(strcmp(temp, "Numrows") == 0 || strcmp(temp, "NumRows") == 0) {
            sscanf(line, "%*s\t:\t%d\n", &numRows);
            break;
        }
    }
    fclose(fp);
    
    if(numRows == 0) {
        sprintf(error_text, "bookshelf_IO: NumRows = %d", numRows);
        runtimeError(error_text);
    }
    
    core_row = (ROW *) malloc((numRows+1)*sizeof(ROW));
    if(!core_row)
        runtimeError("bookshelf_IO: allocation failure in createRowStructure()");
    
    if((fp=fopen(file_name, "r"))==NULL) {
        sprintf(error_text, "bookshelf_IO: Cannot open %s file", file_name);
        runtimeError(error_text);
    }
//    printf("Reading %s ...\n", sclFile);
    
    row_count = 0;
    
    while(!feof(fp)) {
        *line = '\0';
        fgets(line, LINESIZE, fp);
        sscanf(line, "%s\t%*s\n", temp);
        
        if(strlen(line) < 5 || temp[0] == '#' || strcmp(temp, "UCLA") == 0)
            continue;
        
        if(strcmp(temp, "Numrows") == 0 || strcmp(temp, "NumRows") == 0)
            continue;
        
        if(strcmp(temp, "CoreRow") == 0) {
            subrow_count = 0;
            strcpy(core_row[row_count].siteOrient, "N");    // default value for siteorient
            strcpy(core_row[row_count].siteSymmetry, "");   // default value for sitesymmetry
            core_row[row_count].height = 0.0;               // default value for height of core_row
            core_row[row_count].siteWidth = -100.0;
            do {
                *line = '\0';
                fgets(line, LINESIZE, fp);
                sscanf(line, "%s\t:%*s\n", temp);
                
                if(temp[0] == '#') continue;
                
                if(strcmp(temp, "Coordinate")==0) {
                    sscanf(line, "%*s\t:\t%f\n", &core_row[row_count].y_low);
                } else if(strcmp(temp, "Height")==0) {
                    sscanf(line, "%*s\t:\t%f\n", &core_row[row_count].height);
                } else if(strcmp(temp, "Sitewidth")==0) {
                    sscanf(line, "%*s\t:\t%f\n", &core_row[row_count].siteWidth);
                } else if(strcmp(temp, "Sitespacing")==0) {
                    sscanf(line, "%*s\t:\t%f\n", &core_row[row_count].siteSpacing);
                } else if(strcmp(temp, "Siteorient")==0) {
                    sscanf(line, "%*s\t:\t%s\n", core_row[row_count].siteOrient);
                } else if(strcmp(temp, "Sitesymmetry")==0) {
                    sscanf(line, "%*s\t:\t%s\n", core_row[row_count].siteSymmetry);
                } else if(strcmp(temp, "SubrowOrigin")==0) {
                    if(subrow_count > 99) {
                        printf("Row (%d) at coordinate (%.2f) has more than 100 subrows \n", 
                                row_count, core_row[row_count].y_low);
                        printf("Please decrease the number of subrows by \n");
                        printf("creating dummy fixed nodes for the row blockages\n");
                        runtimeError("bookshelf_IO: threshold reached for #subrows per row");
                    }
                    sscanf(line, "%*s\t:\t%f\t%*s\t:\t%d\n", &subroworigin, &totalSites);
                    core_row[row_count].subrow_origin[subrow_count] = subroworigin;
                    core_row[row_count].subrow_end[subrow_count] = subroworigin + totalSites*core_row[row_count].siteSpacing;
                    subrow_count++;
                } else if(strcmp(temp, "End")==0) {
                    break;
                } else {
                    sprintf(error_text, "bookshelf_IO: Unknown parameter: %s", temp);
                    runtimeError(error_text);
                }
            } while(1);
            
            core_row[row_count].subrow_count = subrow_count;
            core_row[row_count].x_low  = core_row[row_count].subrow_origin[0];
            core_row[row_count].x_high = core_row[row_count].subrow_end[subrow_count-1];
            if(core_row[row_count].siteWidth < 0.0) 
                core_row[row_count].siteWidth = core_row[row_count].siteSpacing;
            
            row_count++;
        }
    }
    
    siteOriginX = core_row[0].x_low;
    siteEndX    = core_row[0].x_high;
    for(j=1; j<row_count; j++) {
        if(core_row[j].x_low < siteOriginX)
            siteOriginX = core_row[j].x_low;
        if(core_row[j].x_high > siteEndX)
            siteEndX = core_row[j].x_high;
    }
    siteOriginY = core_row[0].y_low;
    siteEndY    = core_row[row_count-1].y_low + core_row[row_count-1].height;  // possible that height=0 for some rows in Bookshelf 
    
    coreRowHeight = (siteEndY-siteOriginY)/numRows;
    coreHeight = siteEndY - siteOriginY;            // height of placement area 
    coreWidth = siteEndX - siteOriginX;             // width of placement area
    
    
    // getting row blockage related information
    rowBlockageCount = 0;
    for(j=0; j<row_count; j++) {
        // (a) multiple blockages in the row or (b) blockage is within the row
        if(core_row[j].subrow_count > 1) {
            subrow_count = core_row[j].subrow_count;
            if(core_row[j].subrow_origin[0] > siteOriginX) {
                rowBlockageCount++;
            }
            
            if(core_row[j].subrow_end[subrow_count-1] < siteEndX) {
                rowBlockageCount++;
            }
            
            for(k=1; k<subrow_count; k++) {
                rowBlockageCount++;
            }
        // blockage is only at the ends of the row
        } else {
            // blockage is at left end of the row
            if(core_row[j].x_low > siteOriginX) {
                rowBlockageCount++;
            }
            
            // blockage is at right end of the row
            if(core_row[j].x_high < siteEndX) {
                rowBlockageCount++;
            }
        }
    }
    
    xRowBlockage     = vector(1, rowBlockageCount+1);
    yRowBlockage     = vector(1, rowBlockageCount+1);
    widthRowBlockage = vector(1, rowBlockageCount+1);
    
    numRowBlockages = 0;
    if(rowBlockageCount > 0) {
        for(j=0; j<row_count; j++) {
            // (a) multiple blockages in the row or (b) blockage is within the row
            if(core_row[j].subrow_count > 1) {
                subrow_count = core_row[j].subrow_count;
                if(core_row[j].subrow_origin[0] > siteOriginX) {
                    numRowBlockages++;
                    xRowBlockage[numRowBlockages] = siteOriginX + 0.5*(core_row[j].subrow_origin[0] - siteOriginX);
                    yRowBlockage[numRowBlockages] = core_row[j].y_low + 0.5*core_row[j].height;
                    widthRowBlockage[numRowBlockages] = core_row[j].subrow_origin[0] - siteOriginX;
                }
                
                if(core_row[j].subrow_end[subrow_count-1] < siteEndX) {
                    numRowBlockages++;
                    xRowBlockage[numRowBlockages] = core_row[j].subrow_end[subrow_count-1] + 
                                                    0.5*(siteEndX - core_row[j].subrow_end[subrow_count-1]);
                    yRowBlockage[numRowBlockages] = core_row[j].y_low + 0.5*core_row[j].height;
                    widthRowBlockage[numRowBlockages] = siteEndX - core_row[j].subrow_end[subrow_count-1];
                }
                
                for(k=1; k<subrow_count; k++) {
                    numRowBlockages++;
                    xRowBlockage[numRowBlockages] = core_row[j].subrow_end[k-1] + 
                                                    0.5*(core_row[j].subrow_origin[k] - core_row[j].subrow_end[k-1]);
                    yRowBlockage[numRowBlockages] = core_row[j].y_low + 0.5*core_row[j].height;
                    widthRowBlockage[numRowBlockages] = core_row[j].subrow_origin[k] - core_row[j].subrow_end[k-1];
                }
            // blockage is only at the ends of the row
            } else {
                // blockage is at left end of the row
                if(core_row[j].x_low > siteOriginX) {
                    numRowBlockages++;
                    xRowBlockage[numRowBlockages] = siteOriginX + 0.5*(core_row[j].x_low - siteOriginX);
                    yRowBlockage[numRowBlockages] = core_row[j].y_low + 0.5*core_row[j].height;
                    widthRowBlockage[numRowBlockages] = core_row[j].x_low - siteOriginX;
                }
                
                // blockage is at right end of the row
                if(core_row[j].x_high < siteEndX) {
                    numRowBlockages++;
                    xRowBlockage[numRowBlockages] = core_row[j].x_high + 0.5*(siteEndX - core_row[j].x_high);
                    yRowBlockage[numRowBlockages] = core_row[j].y_low + 0.5*core_row[j].height;
                    widthRowBlockage[numRowBlockages] = siteEndX - core_row[j].x_high;
                }
            }
        }
        
        if(numRowBlockages != rowBlockageCount) {
            runtimeError("bookshelf_IO: error during createRowBlockage()");
        }
    }
    
    
    // getting the chip dimensions (if there are perimeter IOs)
    maxX = siteEndX;
    minX = siteOriginX;
    maxY = siteEndY;
    minY = siteOriginY;
    for(j=movableNodes+1; j<=numNodes; j++) {
        if(xCellCoord[j] > maxX) maxX = xCellCoord[j];
        else if(xCellCoord[j] < minX) minX = xCellCoord[j];
        
        if(yCellCoord[j] > maxY) maxY = yCellCoord[j];
        else if(yCellCoord[j] < minY) minY = yCellCoord[j];
    }
    
    fclose(fp);
    if(row_count != numRows) {
        sprintf(error_text, "bookshelf_IO: NumRows (%d) != Number of Row descriptions (%d)", numRows, row_count);
        runtimeError(error_text);
    }
}


/* -----------------------------------------------------------
   Creates a hash linking netname to netNo
   Reads the .wts file to get net-weights
   
   creates extern vars:
      netWeight[]
----------------------------------------------------------- */
void readWtsFile(char benchmarkPath[], char wtsFile[], int net_weights)
{
    FILE *fp;
    char file_name[BUFFERSIZE], temp[BUFFERSIZE];
    char line[LINESIZE];
    float weight;
    long netNo, j;
    long R, hashfunc, RN_index;
    
    
    if(netWeight == NULL) {
        netWeight = vector(1, numNets);
        for(netNo=1; netNo<=numNets; netNo++)
            netWeight[netNo] = 1.0;
    } else {
        runtimeError("bookshelf_IO: allocation error in readWtsFile()");
    }
    
    if(net_weights) {
        
        // defining hash variables
        hashBits = 3+(long)(log(numNets)/log(2));
        hashSize = hashsize(hashBits);
        
        if(NetsInfo == NULL) {
            NetsInfo = (NODES **) malloc(hashSize*sizeof(NODES *));
            if(!NetsInfo)
                runtimeError("bookshelf_IO: allocation failure in createNetsHash()");
        } else {
            runtimeError("bookshelf_IO: NetsHash is not NULL");
        }
        for(j=0; j<hashSize; j++)
            NetsInfo[j] = NULL;
        
        if(RN == NULL && hashFlag == NULL) {
            RN = lvector(1, hashSize);
            hashFlag = cvector(1, hashSize);
        } else {
            runtimeError("bookshelf_IO: error while creating nets hash");
        }
        
        // initialize hash flags
        for(j=1; j<=hashSize; j++)
            hashFlag[j] = 0;
        
        // generate random sequence
        R = 1;
        for(j=1; j<=hashSize; j++) {
            R = ((5*R) & hashmask(hashBits));
            RN[j] = R/4;
        }
        
        // create a hash table for name searching
        for(netNo=1; netNo<=numNets; netNo++) {
            
            hashfunc = (long)get_hashfunc(netName[netNo], 
                                          strlen(netName[netNo])*sizeof(char), 
                                          13);
            hashfunc = (hashfunc & hashmask(hashBits));
            
            RN_index = 1;
            while(hashFlag[hashfunc] != 0 && RN_index < hashSize) {
                hashfunc = ((hashfunc + RN[RN_index]) & hashmask(hashBits));
                RN_index++;
            }
            
            if(hashfunc >= hashSize) {
                sprintf(error_text, "bookshelf_IO: Cannot fill %s in hash table", netName[netNo]);
                runtimeError(error_text);
            }
            
            NetsInfo[hashfunc] = (NODES *) malloc(sizeof(NODES));
            NetsInfo[hashfunc]->name  = netName[netNo];
            NetsInfo[hashfunc]->index = netNo;
            hashFlag[hashfunc] = 1;
        }
        
        strcpy(file_name, benchmarkPath);
        strcat(file_name, "/");
        strcat(file_name, wtsFile);
        
        if((fp=fopen(file_name, "r")) == NULL) {
            sprintf(error_text, "bookshelf_IO: Cannot open: %s file", file_name);
            runtimeError(error_text);
        }
//        printf("Reading %s ...\n", wtsFile);
        
        while(!feof(fp)) {
            *line = '\0';
            fgets(line, LINESIZE, fp);
            sscanf(line, "%s\t%*s\n", temp);
            
            if(strlen(line) < 5 || temp[0] == '#' || strcmp(temp, "UCLA") == 0)
                continue;
            
            sscanf(line, "%s%f\n", temp, &weight);
            
            // find the netNo corresponding to temp
            hashfunc = (long)get_hashfunc(temp, strlen(temp)*sizeof(char), 13);
            hashfunc = (hashfunc & hashmask(hashBits));
            
            RN_index = 1;
            while(RN_index < hashSize) {
                if(NetsInfo[hashfunc] && strcmp(temp, NetsInfo[hashfunc]->name) == 0) {
                    break;
                } else {
                    hashfunc = ((hashfunc + RN[RN_index]) & hashmask(hashBits));
                    RN_index++;
                }
            }
            
            if(RN_index >= hashSize) {
                sprintf(error_text, "bookshelf_IO: Cannot find %s in hash table", temp);
                runtimeError(error_text);
            }
            
            netNo = NetsInfo[hashfunc]->index;
            
            // store netWeight corresponding to netNo
            netWeight[netNo] = weight;
        }
        
        fclose(fp);
        
        for(j=0; j<hashSize; j++) {
            if(NetsInfo[j]) {
                free(NetsInfo[j]);
                NetsInfo[j] = NULL;
            }
        }
        free(NetsInfo);
        NetsInfo = NULL;
        
        free_lvector(RN, 1, hashSize);
        RN = NULL;
        
        free_cvector(hashFlag, 1, hashSize);
        hashFlag = NULL;
    }
}


/* -----------------------------------------------------------
   writes out a plain bookshelf format .pl file
----------------------------------------------------------- */
void writePlFile(char outputDir[], char fileName[], float xCoord[], float yCoord[], 
                 int totalNodes) 
{
    FILE *fp;
    char tempStr[BUFFERSIZE];
    int i;


    strcpy(tempStr, outputDir);
    strcat(tempStr, "/");
    strcat(tempStr, fileName);
    
    if( (fp=fopen(tempStr,"w")) == NULL ) {
        sprintf(error_text, "bookshelf_IO: Cannot open: %s file for write", tempStr);
        runtimeError(error_text);
    }
    printf("\nPrinting %s File... \n",tempStr);

    fprintf(fp, "UCLA pl 1.0\n");

    for(i=1; i<=totalNodes; i++)
        fprintf(fp, "    %20s    %-10f    %-10f  :  %4s\n", 
                cellName[i], xCoord[i]-0.5*cellWidth[i], 
                yCoord[i]-0.5*cellHeight[i], cellOrientation[i]);
    
    fclose(fp);
}

