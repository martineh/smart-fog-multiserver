/*===============================================================================

  Project: SmartFog
  File: xml-utils.hpp

  Copyright (c) 2021, University of Córdoba, The Advanced Informatics Research Group (GIIA).

  All rights reserved. This is propietary software. In no event shall the author
  be liable for any claim or damages.

===============================================================================*/
#ifndef __XML_UTILS_h
#define __XML_UTILS_h

#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <string.h>
#include <netinet/in.h>
#include <queue>
#include <iostream>
#include <unistd.h>

#include "tinyxml2.h"

#define PACKAGE_SIZE 512
#define MAX_PACK 48

#define XML_FILE      "../config/XMLConfig.xml"

using namespace tinyxml2;
using namespace std;

typedef struct xmlConfig {
  char ip[20];
  int port;
  int numRows;
  int numRowsReduction;
  int numCols;
  int numColsReduction;
  int fps;
  int inColor;
  int colorReduction;
  int bAverage;
  int pxThreshold;
  int imgThreshold;
} xmlConfig_t;

xmlConfig_t * newXmlConfig();

void freeXmlConfig(xmlConfig_t * xmlConfig);

xmlConfig_t * initXmlConfig(char *ip, int port, int numRows,
			    int numRowsReduction, int numCols,
			    int numColsReduction, int fps,
			    int inColor, int colorReduction,
			    int bAverage, int pxThreshold,
			    int imgThreshold, xmlConfig_t *xmlConfig);

int readXmlConfig(const char *inputFile, xmlConfig_t *xmlConfig);

void writeXmlConfig(const char *outputFile, xmlConfig_t *xmlConfig);

void printXmlConfig(xmlConfig_t *xmlConfig);

void packXmlConfig(char *dataPack, xmlConfig_t *xmlConfig);

void unpackXmlConfig(char *dataPack, xmlConfig_t *xmlConfig);

bool equalXmlConfig(xmlConfig_t *xmlReconfig, xmlConfig_t *xmlConfig);

#endif
