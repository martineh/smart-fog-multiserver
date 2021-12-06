#include "camera-utils.hpp"

using namespace std;
using namespace cv;

/****   D E F I N E    S E C T I O N   *******/
#define NUM_THREADS 3
#define NUM_CPUS    4

#define TH_PRODUCER 0
#define TH_WORKER   1
#define TH_CONSUMER 2

#define XML_CONFIG_FILE    "../xml-config.xml"

/**** G L O B A L    V A R I A B L E S *******/
//Global Queues
queue<Mat> q_capture;
queue<Mat> q_send;

//Global Mutex
pthread_mutex_t m;
pthread_mutex_t m_send;

int items;
int items_send;

bool finished;

//GLOBAL VARIABLE FOR DEBUG (PATH TO THE IMAGE)
char *img_path;
bool debug;

/***************   M A I N  *****************/
int main (int argc, char *argv[]) {

  pthread_t threads[NUM_THREADS];
  pthread_attr_t attr;
  cpu_set_t cpu_set;
  int ret;
  
  items = 0;
  finished = false;
  if (argc > 1) {
    debug = true;
    img_path = strdup(argv[1]);
  }

  //Read XML config
  xmlConfig_t * xmlConfig = newXmlConfig();
  readXmlConfig(XML_CONFIG_FILE, xmlConfig);
  printXmlConfig(xmlConfig);

  std::cout << "                                          " << std::endl;
  std::cout << "------------------------------------------" << std::endl;
  std::cout << "              [SCHEME LEVEL]              " << std::endl;
  std::cout << "------------------------------------------" << std::endl;
  std::cout << "       [Raspberry Pi (C++)]-->(Py)        " << std::endl;
  std::cout << "                                          " << std::endl;
  std::cout << "             [CAMERA CAPTURE]             " << std::endl;
  std::cout << "******************************************" << std::endl;
   std::cout << "                                          " << std::endl;
   
  if (pthread_mutex_init(&m, NULL) != 0) {
    std::cout << "Mutex init failed." << std::endl;
    return 0;
  }

  items_send = 0;
  if (pthread_mutex_init(&m_send, NULL) != 0) {
    std::cout << "Mutex init failed." << std::endl;
    return 0;
  }

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  //Thread creation, assign a handler to each thread and mapped each thread to each CPU id
  for (int i = 0; i < NUM_THREADS; i++) {
    CPU_ZERO(&cpu_set);
    CPU_SET(i, &cpu_set);
    sched_setaffinity(syscall(SYS_gettid), sizeof(cpu_set), &cpu_set);

    if (i == TH_PRODUCER) {
      ret = pthread_create(&threads[i], &attr, getFrame, (void *)xmlConfig);
    } else if (i == TH_CONSUMER) {
      ret = pthread_create(&threads[i], &attr, processFrame, (void *)xmlConfig);
    } else {
      ret = pthread_create(&threads[i], &attr, sendFrame, (void *)xmlConfig);
    }
    
    if (ret) {
      std::cout << "Error: Unable to create thread producer." << std::endl;
      return 0;
    }
    
  }

  //Thread destructor
  void *status;
  pthread_attr_destroy(&attr);
  for (int i = 0; i < NUM_THREADS; i++) {
    if ((ret = pthread_join(threads[i], &status))) {
      std::cout << "ERROR; return code from pthread_join() is " << ret << std::endl;
      return 0;
    }
  }
  
  return 0;
  
}
