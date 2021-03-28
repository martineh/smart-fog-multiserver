#include "camera-utils.hpp" 

/****   D E F I N E    S E C T I O N   *******/
//* For the Camera 

//* For the TCP/IP Conection
#define PORT 5050
#define SA struct sockaddr 
#define MAX_UDP_PACK 65000

//* For the Queue
#define QUEUE_MAX_FRAMES   10

//* For Threads
#define BACKGROUND_AVERAGE 30
#define THRESHOLD_PX       5
#define THRESHOLD_SEND     5

#define ERROR_CODE   0
#define LOG_CODE     1
#define WARNING_CODE 2
#define TIMING_CODE  3
#define OK_CODE      4

/**** G L O B A L    V A R I A B L E S *******/
extern queue<Mat> q_capture;
extern queue<Mat> q_send;

extern pthread_mutex_t m;
extern pthread_mutex_t m_send;

extern int items;
extern int items_send;

extern bool finished;

extern char *date_str;

//GLOBAL VARIABLE FOR DEBUG (PATH TO THE IMAGE)
extern char *img_path;
extern bool debug;

/*****  T I M I N G    F U N C T I O N  *****/
void timerStart(timeval *t_start) {
  gettimeofday(t_start, NULL);
}


void timerStop(timeval *t_stop) {
  gettimeofday(t_stop, NULL);
}


double getTime(timeval t_start, timeval t_stop) {
  return   ((t_stop.tv_sec - t_start.tv_sec) + ((t_stop.tv_usec - t_start.tv_usec)/1000000.0));
}


char *getDateAndTime(char *date_str) {
  std::time_t now = std::time(0);
  tm *ltm = localtime(&now);
  sprintf(date_str, "[%02d/%02d/%d %02d:%02d:%02d]", ltm->tm_mday, 1 + ltm->tm_mon, 1900 + ltm->tm_year,
	  ltm->tm_hour,ltm->tm_min,ltm->tm_sec);

  return date_str;  

}

char *getLogCode(int code, char *code_str) {

  switch(code) {
      case ERROR_CODE:
	sprintf(code_str, "%s", "[\x1B[31mERROR\033[0m]");
	break;
      case LOG_CODE:
	sprintf(code_str, "%s", "[\x1B[93mLOG\033[0m]");
	break;
      case WARNING_CODE:
	sprintf(code_str, "%s", "[\x1B[33mWARNING\033[0m]");
	break;
      case TIMING_CODE:
	sprintf(code_str, "%s", "[\x1B[94mTIMING\033[0m]");
	break;
      case OK_CODE:
	sprintf(code_str, "%s", "[\x1B[92mOK\033[0m]");
	break;

  }

  return code_str;
  
}

char *getPrintMsg(int code, char *log_msg) {
  char date_str[128];
  char code_str[64];
  
  getLogCode(code, code_str);
  getDateAndTime(date_str);
  
  sprintf(log_msg, "%s%s ", code_str, date_str);

  return log_msg;
}

/*****  Q U E U E    F U N C T I O N  *****/
void insert_capture(Mat img) {
  pthread_mutex_lock(&m);
  if (items < QUEUE_MAX_FRAMES) {
    q_capture.push(img);
    items++;
  } else {
    std::cout << "Queue max:" << items << std::endl;
  }
  pthread_mutex_unlock(&m);
}

Mat pop_capture() {
  Mat img;

  while (img.empty()) {
    pthread_mutex_lock(&m);
    if (items > 0) {
      img = q_capture.front();
      q_capture.pop();
      items--;
    } 
    pthread_mutex_unlock(&m);
  }
  
  return img;  
}

void insert_send(Mat img) {
  pthread_mutex_lock(&m_send);
  if (items < QUEUE_MAX_FRAMES) {
    q_send.push(img);
    items_send++;
  } else {
    std::cout << "Queue max:" << items << std::endl;
  }
  pthread_mutex_unlock(&m_send);
}

Mat pop_send() {
  Mat img;

  while (img.empty()) {
    pthread_mutex_lock(&m_send);
    if (items_send > 0) {
      img = q_send.front();
      q_send.pop();
      items_send--;
    } 
    pthread_mutex_unlock(&m_send);
  }
  
  return img;  
}


/*****  I M A G E    F U N C T I O N  *****/
unsigned char *newImageVectorPack(Mat img, unsigned int *max_buf) {
  //==== Vector Size and Bits Data Position: =====//
  //   2 bytes (rows)     +                       //
  //   2 bytes (columns)  +                       //
  //   1 byte  (channels) +                       //
  //   1 byte  (depth)    +                       //
  //   n bytes (Pixels Rows x Columns x Channels) //
  //==============================================//
  short rows             = img.rows;
  short cols             = img.cols;
  unsigned char channels = img.channels();
  unsigned char depth    = img.type() & CV_MAT_DEPTH_MASK;

  //Buffer New
  *max_buf = (rows * cols * channels) + 6;
  unsigned char *v_colors = new unsigned char[*max_buf];
  unsigned char *v_colors_tmp;
  
  v_colors_tmp = v_colors;

  //Copy the image parameters into the vector 
  memcpy(v_colors_tmp, &rows, sizeof rows);
  v_colors_tmp += (sizeof rows);
  
  memcpy(v_colors_tmp, &cols, sizeof cols);
  v_colors_tmp += (sizeof cols);

  memcpy(v_colors_tmp, &channels, sizeof channels);
  v_colors_tmp += (sizeof channels);

  memcpy(v_colors_tmp, &depth, sizeof depth);
  v_colors_tmp += (sizeof depth);

  //Copy the image into the vector
  unsigned int v_index = 0;
  if (channels == 3) {
    //Image with three channels
    for (short i = 0; i < rows; i++) {
      for (short j = 0; j < cols; j++) {
    	for (unsigned char c = 0; c < channels; c++) {
    	  v_colors_tmp[v_index++] = (unsigned char)img.at<Vec3b>(i,j)[c];
    	}
      }
    }
  } else {
    //Image with one channel
    for (short i = 0; i < rows; i++) {
      for (short j = 0; j < cols; j++) {
	v_colors_tmp[v_index++] = (unsigned char)img.at<uchar>(i,j);
      }
    }  
  }
  
  return v_colors;
  
}

//Calculate  Average Background
Mat calculateBackground(unsigned int average) {
  unsigned int n_average = 0;
  Mat img, background;
  Mat acum_blue, acum_green, acum_red;
  
  while(n_average < average) {    
    img = pop_capture();
    
    if (n_average == 0) {
      background = img.clone();
      acum_blue  = Mat(img.rows, img.cols, CV_32F);
      acum_green = Mat(img.rows, img.cols, CV_32F);
      acum_red   = Mat(img.rows, img.cols, CV_32F);
    }
    
    for (int i = 0; i < img.rows; i++) {
      for (int j = 0; j < img.cols; j++) {
	acum_blue.at<float>(i,j) += (float)img.at<Vec3b>(i,j)[0];
	acum_green.at<float>(i,j)+= (float)img.at<Vec3b>(i,j)[1];
	acum_red.at<float>(i,j)  += (float)img.at<Vec3b>(i,j)[2];
      }
    }      	
    n_average++;
    img.release();
  }
  
  //Values Average
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      background.at<Vec3b>(i, j)[0] = (unsigned char)(acum_blue.at<float>(i,j) / n_average);
      background.at<Vec3b>(i, j)[1] = (unsigned char)(acum_green.at<float>(i,j) / n_average);
      background.at<Vec3b>(i, j)[2] = (unsigned char)(acum_red.at<float>(i,j) / n_average);
    }
  }

  acum_blue.release();
  acum_green.release();
  acum_red.release();
  
  return background;  
}

int differenceRatioBackground(Mat background, Mat img, int threshold_pixel) {
  int different_px = 0;
  int max_distance = 255 + 255 + 255; // Red + Green + Blue
  
  //Different pixels between the new image "img" and the background "background"
  for (int i = 0; i < background.rows; i++) {
    for (int j = 0; j < background.cols; j++) {
      Vec3b pixels_a = img.at<Vec3b>(i, j);
      Vec3b pixels_b = background.at<Vec3b>(i, j);
      unsigned int distance = abs(pixels_a[0] - pixels_b[0]) + 
	abs(pixels_a[1] - pixels_b[1]) +
	abs(pixels_a[2] - pixels_b[2]);
      int distance_ratio  = (distance * 100) / max_distance;
      if (distance_ratio > threshold_pixel)
	different_px += 1;
    }
  }      

  //Calculate and return the difference ratio 
  return ((different_px * 100) / (background.rows * background.cols));
}

// Bits reduction for image with 3 channels
Mat colorImageReduction(Mat img, int numBits) {
  uchar maskBit = 0xFF;

  maskBit = maskBit << (8 - numBits);

  switch(img.channels()) {
  case 1: //One channel
    {
      for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
          img.at<uchar>(i, j) = img.at<uchar>(i, j) & maskBit;
        }
      }
    }
  case 3: //Three channels
    {
      for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
          Vec3b valVec = img.at<Vec3b>(i, j);
          valVec[0] = (valVec[0] & maskBit);
          valVec[1] = (valVec[1] & maskBit);
          valVec[2] = (valVec[2] & maskBit);
          img.at<Vec3b>(i, j) = valVec;
        }
      }
    }
  default:
    {
      for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
          Vec3b valVec = img.at<Vec3b>(i, j);
          valVec[0] = (valVec[0] & maskBit);
          valVec[1] = (valVec[1] & maskBit);
          valVec[2] = (valVec[2] & maskBit);
          img.at<Vec3b>(i, j) = valVec;
        }
      }
    }
  }
  
  return img;
}

// Change image resolution with the same image size
//TODO: Memory leak, WTF?
Mat changeImageResolution(Mat img, int rows, int cols) {
  Mat outImage;
  Mat resizedImage;
  Size newSize(rows, cols);
  Size OriginalSize(img.cols, img.rows);
  
  resize(img, resizedImage, newSize);//resize image
  resize(resizedImage, outImage, OriginalSize);//resize image

  resizedImage.release();
  img.release();
  
  return outImage;
}

Mat imageTransformHandler(Mat img, int colorReduction, 
			  int rowsReduction, int colsReduction,
			  bool enableTransform) {
  if (enableTransform) {
    if (colorReduction != 1)
      img = colorImageReduction(img, colorReduction);
    
    if((img.rows != rowsReduction) ||
       (img.cols != colsReduction)) {
      //img = changeImageResolution(img, rowsReduction, colsReduction);
    }
  }
  
  return img;
}

/*****  T H R E A D S    F U N C T I O N  *****/

//= C A P T U R E    F R A M E S =//
void *getFrame(void *input) {

  if (debug) {
    std::cout << "IMAGE PATH:" << img_path << std::endl;
    Mat img = imread(img_path, IMREAD_COLOR);
    insert_capture(img);

    finished = true;
    return NULL;
  }
  
  xmlConfig_t *xmlConfig = (xmlConfig_t *)input;
  VideoCapture cap(0, cv::CAP_V4L); //VideoCapture From 0
  Mat img;
  
  if(!cap.isOpened()) {
    std::cout << "[ERROR] I can't open the camera. Exit." << std::endl;
    exit(-1);
  }

  //Image Size Fixed
  cap.set(cv::CAP_PROP_FRAME_WIDTH,  xmlConfig->numRows);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, xmlConfig->numCols);
  
  while (true) {
    cap >> img;
    insert_capture(img);//img.clone());
  }

  finished = true;
  return NULL; 
}


//= P R O C E S S    F R A M E S =//
void *processFrame(void *input) {

  if (debug) {
    Mat img = pop_capture();
    insert_send(img);
    return NULL;
  }

  xmlConfig_t *xmlConfig = (xmlConfig_t *)input;
  Mat img, background;
  
  int colorReduction = xmlConfig->colorReduction;
  int rowsReduction  = xmlConfig->numRowsReduction;
  int colsReduction  = xmlConfig->numColsReduction;
  
  background = calculateBackground(xmlConfig->bAverage);
  
  while(!finished) {
    img = pop_capture();
    int threshold = differenceRatioBackground(background, img, xmlConfig->pxThreshold);
    if (threshold > xmlConfig->imgThreshold || true) {
      img = imageTransformHandler(img, colorReduction, rowsReduction, colsReduction, true);
      insert_send(img);
    }
      //img.release();
  }
  
  return NULL;  
}


//= S E N D    F R A M E S =// 
void *sendFrame(void *input) {  

  xmlConfig_t *xmlConfig = (xmlConfig_t *)input;
  int sockfd; 
  struct sockaddr_in servaddr;
  
  char log_str[256];
  char code_str[128];

  size_t n_frames = 0;
  timeval t_start, t_stop;
  
  // socket create and varification 
  sockfd = socket(AF_INET, SOCK_STREAM, 0); 
  if (sockfd == -1) {
    std::cout << getPrintMsg(ERROR_CODE, log_str) << "Connecting with the server failed." << std::endl;
    exit(-1); 
  } 
  
  bzero(&servaddr, sizeof(servaddr));   
  servaddr.sin_family = AF_INET;   
  servaddr.sin_addr.s_addr = inet_addr(xmlConfig->ip); //Mi PC
  servaddr.sin_port = htons(xmlConfig->port);
  
  // connect the client socket to server socket 
  if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) {
    std::cout << getPrintMsg(ERROR_CODE, log_str) << "Connecting with the server failed." << std::endl;
    exit(-1); 
  } 

  Mat img;
  unsigned int max_buff;
  unsigned char *imgPackBuff;

  std::cout <<  getPrintMsg(LOG_CODE, log_str) << "CLIENT started on " <<
    xmlConfig->ip << ":" << xmlConfig->port << " " << getLogCode(OK_CODE, code_str) << std::endl;

  timerStart(&t_start);
  while(!finished) {
    img = pop_send();

    imgPackBuff = newImageVectorPack(img, &max_buff);
    //if (debug)
    //std::cout << "[CLIENT] Send image of " << max_buff << "(bytes)" << std::endl;    
    size_t b = write(sockfd, imgPackBuff, max_buff);
    //sleep(30);
    if (b == 0)
      std::cout << "[WARNING] No data writed to the server." << std::endl;

    free(imgPackBuff);
    img.release();

    n_frames += 1;
    if ((n_frames % 10) == 0) {
        timerStop(&t_stop);
	double t_total = getTime(t_start, t_stop);
	printf("%sCLIENT %d frames captured in %0.2f(s) [%0.2f(fps)]\r",
	       getPrintMsg(TIMING_CODE, log_str), n_frames, t_total, n_frames / t_total);
	fflush(stdout);
    }    
  }
  
  close(sockfd);
  
  return NULL;
}
