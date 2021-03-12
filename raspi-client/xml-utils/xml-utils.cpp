/*===============================================================================

  Project: SmartFog
  File: xmlConfig.cpp

  Copyright (c) 2020, University of Córdoba, The Advanced Informatics Research Group (GIIA).

  All rights reserved. This is propietary software. In no event shall the author
  be liable for any claim or damages.

===============================================================================*/

#include "xml-utils.hpp"

xmlConfig_t * newXmlConfig()
{
  xmlConfig_t *xmlConfig = (xmlConfig_t *) malloc(sizeof(xmlConfig_t));

  return xmlConfig;  
}

void freeXmlConfig(xmlConfig_t * xmlConfig)
{
  free(xmlConfig);
}

xmlConfig_t * initXmlConfig(char *ip, int port, int numRows,
			    int numRowsReduction, int numCols,
			    int numColsReduction, int fps,
			    int inColor, int colorReduction,
			    int bAverage, int pxThreshold,
			    int imgThreshold, xmlConfig_t *xmlConfig)
{  
  strcpy(xmlConfig->ip, ip);
  xmlConfig->port             = port;
  xmlConfig->numRows          = numRows;
  xmlConfig->numRowsReduction = numRowsReduction;
  xmlConfig->numCols          = numCols;
  xmlConfig->numColsReduction = numColsReduction;
  xmlConfig->fps              = fps;
  xmlConfig->inColor          = inColor;
  xmlConfig->colorReduction   = colorReduction;
  xmlConfig->bAverage         = bAverage;
  xmlConfig->pxThreshold      = pxThreshold;
  xmlConfig->imgThreshold     = imgThreshold;
  
  return xmlConfig;
}



int readXmlConfig(const char *inputFile, xmlConfig_t *xmlConfig)
{
  
  XMLDocument xmlDocRead;   // XML Document
  XMLError eResultRead;     // Catch Errors
  XMLNode *pRootRead;       // Father Node XML
  XMLElement *pElementRead; // Sons Nodes
  
  const char *resolution, *resolutionReduce;
  char ip[20];
  int port, numRows, numRowsReduction,
    numCols, numColsReduction, fps,
    colorReduction, inColor, bAverage,
    pxThreshold, imgThreshold;
  
  //XML File reading
  eResultRead = xmlDocRead.LoadFile(inputFile);
  
  //Catch Errors
  XMLCheckResult(eResultRead); 
  
  //Father Node Read
  pRootRead = xmlDocRead.FirstChild(); 
  if (pRootRead == nullptr)
  {
    std::cout<<"Error al leer el nodo padre.\n";
    return XML_ERROR_FILE_READ_ERROR;
  }
  
  //-> GET IP
  pElementRead = pRootRead->FirstChildElement("IpToSend"); 
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"IpToSend\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  strcpy(ip, pElementRead->GetText());
  
  //-> GET PORT
  pElementRead = pRootRead->FirstChildElement("PortToSend"); 
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"PortToSend\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  eResultRead = pElementRead->QueryIntText(&port);
  XMLCheckResult(eResultRead);
  
  //-> GET ROWS X COLS
  pElementRead = pRootRead->FirstChildElement("ResolutionToSend");
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"ResolutionToSend\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  resolution = pElementRead->GetText();
  sscanf(resolution, "%dx%d", &numRows, &numCols);
  
  //-> GET FPS
  pElementRead = pRootRead->FirstChildElement("FramesPerSecond");
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"FramesPerSecond\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  eResultRead = pElementRead->QueryIntText(&fps);
  XMLCheckResult(eResultRead);
  
  //-> GET COLOR
  pElementRead = pRootRead->FirstChildElement("ColorImage");
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"ColorImage\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  eResultRead = pElementRead->QueryIntText(&inColor);
  XMLCheckResult(eResultRead);

  //-> GET ROWS X COLS REDUCTION
  pElementRead = pRootRead->FirstChildElement("ReducedSpatialResolution");
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"ReducedSpatialResolution\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  resolutionReduce = pElementRead->GetText();
  sscanf(resolutionReduce, "%dx%d", &numRowsReduction, &numColsReduction);

  //-> GET COLOR REDUCTION
  pElementRead = pRootRead->FirstChildElement("ReducedColorResolutionInBits");
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"ReducedColorResolutionInBits\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  eResultRead = pElementRead->QueryIntText(&colorReduction);
  XMLCheckResult(eResultRead);

  //-> GET BACKGROUND AVERAGE
  pElementRead = pRootRead->FirstChildElement("BackgroundAverage");
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"BackgroundAverage\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  eResultRead = pElementRead->QueryIntText(&bAverage);
  XMLCheckResult(eResultRead);

  //-> GET PIXEL THRESHOLD
  pElementRead = pRootRead->FirstChildElement("PixelThreshold"); 
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"PixelThreshold\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  eResultRead = pElementRead->QueryIntText(&pxThreshold);
  XMLCheckResult(eResultRead);
  
  //-> GET IMAGE THRESHOLD
  pElementRead = pRootRead->FirstChildElement("ImageThreshold"); 
  if (pElementRead == nullptr)
  {
    std::cout<<"Error al leer el nodo hijo: \"ImageThreshold\".\n";
    return XML_ERROR_PARSING_ELEMENT;
  }
  eResultRead = pElementRead->QueryIntText(&imgThreshold);
  XMLCheckResult(eResultRead);
  
  xmlConfig = initXmlConfig(ip, port, numRows, numRowsReduction,
			    numCols, numColsReduction, fps, inColor,
			    colorReduction, bAverage, pxThreshold,
			    imgThreshold, xmlConfig);

  return XML_SUCCESS;  
}


void writeXmlConfig(const char *outputFile, xmlConfig_t *xmlConfig) {

  ofstream newXmlFile;
  
  newXmlFile.open(outputFile);
  newXmlFile << "<Root>\n";  
  newXmlFile << "    <IpToSend>"<< xmlConfig->ip << "</IpToSend>\n";
  newXmlFile << "    <PortToSend>"<< xmlConfig->port << "</PortToSend>\n";
  newXmlFile << "    <ResolutionToSend>"<< xmlConfig->numRows << "x" << xmlConfig->numCols << "</ResolutionToSend>\n";
  newXmlFile << "    <FramesPerSecond>"<< xmlConfig->fps << "</FramesPerSecond>\n";
  newXmlFile << "    <ColorImage>"<< xmlConfig->inColor << "</ColorImage>\n";
  newXmlFile << "    <ReducedSpatialResolution>"<< xmlConfig->numRowsReduction
	     << "x"<< xmlConfig->numColsReduction << "</ReducedSpatialResolution>\n";
  newXmlFile << "    <ReducedColorResolutionInBits>" << xmlConfig->colorReduction << "</ReducedColorResolutionInBits>\n";
  newXmlFile << "    <BackgroundAverage>" << xmlConfig->bAverage     << "</BackgroundAverage>\n";
  newXmlFile << "    <PixelThreshold>"    << xmlConfig->pxThreshold  << "</PixelThreshold>\n";
  newXmlFile << "    <ImageThreshold>"    << xmlConfig->imgThreshold << "</ImageThreshold>\n";
  
  newXmlFile << "</Root>\n";
  
  newXmlFile.close();
  
}


void printXmlConfig(xmlConfig_t *xmlConfig) {

  std::cout << std::endl;
  std::cout << "******************************************"     << std::endl;
  std::cout << "**  X M L    C O N F I G U R A T I O N  **"     << std::endl;
  std::cout << "******************************************"     << std::endl;
  std::cout << " CONNECTION                               "     << std::endl;
  std::cout << "------------------------------------------"     << std::endl;
  std::cout << "   >IP       : " << xmlConfig->ip               << std::endl;
  std::cout << "   >PORT     : " << xmlConfig->port             << std::endl;
  std::cout << "------------------------------------------"     << std::endl;
  std::cout << " IMAGE CAPTURE                            "     << std::endl;
  std::cout << "------------------------------------------"     << std::endl;
  std::cout << "   >ROWS     : " << xmlConfig->numRows          << std::endl;
  std::cout << "   >COLUMNS  : " << xmlConfig->numCols          << std::endl; 
  std::cout << "   >FPS      : " << xmlConfig->fps              << std::endl;
  std::cout << "   >COLOR    : " << xmlConfig->inColor          << std::endl;
  std::cout << "------------------------------------------"     << std::endl;
  std::cout << " IMAGE REDUCTION                          "     << std::endl;
  std::cout << "------------------------------------------"     << std::endl;
  std::cout << "   >ROWS     : " << xmlConfig->numRowsReduction << std::endl;
  std::cout << "   >COLUMNS  : " << xmlConfig->numColsReduction << std::endl;
  std::cout << "   >COLOR    : " << xmlConfig->colorReduction   << std::endl;
  std::cout << "------------------------------------------"     << std::endl;
  std::cout << " BACKGROUND                               "     << std::endl;
  std::cout << "------------------------------------------"     << std::endl;
  std::cout << "   >AVERAGE BACKGROUND : " << xmlConfig->bAverage     << std::endl;
  std::cout << "   >PIXEL THRESHOLD    : " << xmlConfig->pxThreshold  << std::endl;
  std::cout << "   >IMAGE THRESHOLD    : " << xmlConfig->imgThreshold << std::endl;
  std::cout << "******************************************"           << std::endl;
}

void packXmlConfig(char *dataPack, xmlConfig_t *xmlConfig) {
  
  //*********************************************
  //   B I T S    P A C K    S T R U C T U R E
  //*********************************************
  //Tag:   20x1(char)  8x4(int)    TOTAL
  //Bits:  [16]        [32]     =  [48]

  unsigned char point          = 0;
  const unsigned char sizeType = sizeof(int);
  
  memcpy(&dataPack[point], xmlConfig->ip, 16); //IP (16 bits)
  point += 16;

  memcpy(&dataPack[point], &xmlConfig->port, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->numRows, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->numRowsReduction, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->numCols, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->numColsReduction, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->fps, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->inColor, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->colorReduction, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->bAverage, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->pxThreshold, sizeType);
  point += sizeType;

  memcpy(&dataPack[point], &xmlConfig->imgThreshold, sizeType);
  point += sizeType;

}

void unpackXmlConfig(char *dataPack, xmlConfig_t *xmlConfig) {

  //*********************************************
  //   B I T S    P A C K    S T R U C T U R E
  //*********************************************
  //Tag:   20x1(char)  8x4(int)    TOTAL
  //Bits:  [16]        [32]     =  [48]

  unsigned char point          = 0;
  const unsigned char sizeType = sizeof(int);
  
  memcpy(xmlConfig->ip, &dataPack[point], 16); //IP (16 bits)
  point += 16;

  memcpy(&xmlConfig->port, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->numRows, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->numRowsReduction, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->numCols, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->numColsReduction, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->fps, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->inColor, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->colorReduction, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->bAverage, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->pxThreshold, &dataPack[point], sizeType);
  point += sizeType;

  memcpy(&xmlConfig->imgThreshold, &dataPack[point], sizeType);
  point += sizeType;

}

bool equalXmlConfig(xmlConfig_t *xmlReconfig, xmlConfig_t *xmlConfig)
{

  //if (strcmp(xmlReconfig->ip, xmlConfig->ip) != 0) { return false; }
  //if (xmlReconfig->port != xmlConfig->port) { return false; }

  if (xmlConfig->numRows          != xmlReconfig->numRows)          { return false; }
  if (xmlConfig->numCols          != xmlReconfig->numCols)          { return false; }
  if (xmlConfig->fps              != xmlReconfig->fps)              { return false; }

  if (xmlConfig->numRowsReduction != xmlReconfig->numRowsReduction) { return false; }
  if (xmlConfig->numColsReduction != xmlReconfig->numColsReduction) { return false; }
  if (xmlConfig->colorReduction   != xmlReconfig->colorReduction)   { return false; }

  if (xmlConfig->inColor          != xmlReconfig->inColor)          { return false; }

  return true;
  
}
