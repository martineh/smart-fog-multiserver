#ifndef CAMERA_UTILS_HPP
#define CAMERA_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <sys/time.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "xml-utils/xml-utils.hpp"

using namespace std;
using namespace cv;


void insert_capture(Mat img);
Mat pop_capture();

/** Start timer
 *  @param t_start Struct to store the actual time
 */
void timerStart(timeval *t_start);


/** Finish timer
 *  @param t_stop Struct to store the actual time
 */
void timerStop(timeval *t_stop);


/** Get the total time between 'timerStart' y 'timerStop'.
 *  @param t_start Struct to store the actual time
 *  @param t_stop  Struct to store the actual time
 *  @return Return total time in seconds
 */
double getTime(timeval t_start, timeval t_stop);

/** Calculates the difference between a new image and the background calculated.
 *  @param background The image with the background calculated
 *  @param img The new image to compare with this background
 *  @param threshold A threshold to decide if a pixel is different than other or not (0-100)
 *  @return Return the tinal difference ratio
 */
float differenceRatioBackground(Mat background, Mat img, float threshold);

/** Convert an image in 'Mat' format to a serialized vector of 'unsigned char'.
 *  The first bits from this vector correspond to the config image parameters 
 *  (rows, cols, channels and depth)
 *  @param img Image to convert.
 *  @param len Output parameter. Recived a pointer that stores the maximum vector size in bytes.
 *  @return Final vector 
 */
unsigned char *newImageVectorPack(Mat img, unsigned int *max_buf);


/** Calcula a partir de 'n' imágenes el patrón del fondo que se usará en el resto
 *  del procesamiento.
 *  @param average Número de imágenes que se usarán para calcular el patrón.
 *  @return Imagen resultante.
 */
Mat calculateBackground(unsigned int average);


/** Hilo POSIX encargado de llevar a cabo la captura de los frames. Inicializa la camara, 
 *  y posteriormente va capturando frames de forma continua y almacenandolos en la cola 
 *  de frames. En caso de llenarse esta cola este hilo lector queda bloqueado hasta que 
 *  haya de nuevo espacio.
 *  @param input Variable donde se almacenan los parametros de entrada. En estos momentos es 'NULL'
 */
void *getFrame(void *input);


/** Hilo POSIX encargado de comparar cada una de las nuevas imágenes capturas con el patrón de fondo 
 *  generado por la función 'calculateBackground'. Si la imagen contiene diferencias significantes 
 *  esta se almacena en la cola de envíos para que sean enviadas al host destino.
 *  @param input Variable donde se almacenan los parametros de entrada. En estos momentos es 'NULL'
 */
void *processFrame(void *input);


/** Hilo POSIX encargado de llevar a cabo los envíos de las imagenes seleccionadas y almacenadas 
 *  en la cola de envios.
 *  @param input Variable donde se almacenan los parametros de entrada. En estos momentos es 'NULL'
 */
void *sendFrame(void *input);

Mat colorImageReduction(Mat inImage, int numBits);
Mat changeImageResolution(Mat inImage, int numRows, int numCols);
int differenceRatiobackground(Mat background, Mat img, float threshold);
Mat imageTransformHandler(Mat img,  int colorReduction,
			  int rowsReduction, int colsReduction,
			  bool enableTransform);
#endif
