/*
 ============================================================================
 Name        : harris_v1.0.cu
 Author      : Julio Gonzalez Carranza
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */
/**
 *
  @file harrisHilos.cu
  @brief Implementación de la función principal de harris

  En este documento se encuentra el algoritmo de harris.
  Este algoritmo se encarga de ejecutar las funciones necesarias para determinar
  el número y la localización de las esquinas en una imagen.

  @author Julio González Carranza
  @date 10/03/2017

*/

#include "defines.hpp"
#include <thread>
#include <CImg.h>
#include <mutex>
#define __TIEMPO__KERNELS__ true
#define __TIEMPO__TOTAL__ true
#define __GRADIENTE__ 3
#define __HISTOGRAMA__ CPU
#define __UMBRAL__ CPU
#define __HARRIS__ 4
#define __MAXIMO__NO__
//#define __MAXIMO__CPU__
//#define __GOODPIXELS__CPU__
#define __GOODPIXELS__GPU__
#define __SORT__ GPU
#define __DIRECTORIO__RAIZ__ "/home/julio/universidad/HarrisCornerRAwareEval"
//#define DEMO_IMAGE	"/home/julio/universidad/proyecto/proyecto/HarrisCornerRAwareEval/corridor/0001.bmp"
//#define SRC_IMAGE		"/home/julio/universidad/proyecto/proyecto/HarrisCornerRAwareEval/corridor/"

using namespace std;
using namespace cimg_library;
void GenFileListSorted(char *dirName, char **fileList, int *count)
{
	struct dirent **namelist;
	int n, fileCnt = 0;

	n = scandir(dirName, &namelist, 0, alphasort);
	if (n < 0)
	{
		//printf(" [SMD][ER]Unable to open directory \n");
		*count = 0;
	}
	else
	{
		int iter = 0;

		while (iter < n)
		{
			if((strcmp(namelist[iter]->d_name, ".") != 0) && (strcmp(namelist[iter]->d_name, "..") != 0))
			{
				// process only .bmp files
				if(strstr(namelist[iter]->d_name, ".bmp"))
				{
					fileList[fileCnt] = (char *)malloc(strlen(dirName) + strlen(namelist[iter]->d_name) + 2);
					if(!fileList[fileCnt])
					{
						//printf(" ERROR: malloc failed ! \n");
						return;
					}
		  			strcpy(fileList[fileCnt], dirName);
		  			strcat(fileList[fileCnt], "/");
		  			strcat(fileList[fileCnt], namelist[iter]->d_name);
					printf(" [SMD][OK]File name: %s \n", fileList[fileCnt]);
					free(namelist[iter]);
					fileCnt++;
				}
			}
			iter++;
		}
		*count = fileCnt;
	}
}

void ReadBMP(char* filename, unsigned char* grayData, int &ancho, int &alto)
{

	unsigned char tmp;
	CImg<unsigned char> img(filename);
	alto=img._height;
	ancho=img._width;
	for(int i=0;i<alto;i++){
		for(int j=0;j<ancho;j++){
			tmp=img(j,i,0,0);
			img(j,i,0,0)=img(j,i,0,2);
			img(j,i,0,2)=tmp;
			grayData[i*ancho+j]=(img(j,i,0,0)+img(j,i,0,1)+img(j,i,0,2))/3;
		}
	}
}

int cornerDetector(unsigned char* grayImage, vector<punto> &puntosDeInteres, const int width, const int height, parametros &param, tiempos &t_kernel){

	const unsigned char *input = grayImage;
	const int nPixels = width*height;
	//const int width = ancho;
	//const int height = alto;
	const int stop = nPixels - width - 1;

	#if __TIEMPO__KERNELS__
		float milliseconds = 0;
		cudaEvent_t e_start, e_stop;
		cudaEventCreate(&e_start);
		cudaEventCreate(&e_stop);
	#endif

	const int diff = HARRIS_WINDOW_SIZE + 1;
	int pOutputImage[width*height];

	int despl = (HARRIS_WINDOW_SIZE / 2) * (width + 1);
	int *R = pOutputImage + (HARRIS_WINDOW_SIZE / 2) * (width + 1);

	int max[1];
	int maximum=0;
	max[0]=0;

	cudaMemcpy(param.d_max,max,1*sizeof(int),cudaMemcpyHostToDevice);
	/**
	 * Declaracion de los vectores p, q y pq para accesos coalescentes a memoria
	 * de la gpu
	 *
	 */

	cudaMemcpy(param.d_input,input,nPixels*sizeof(unsigned char),cudaMemcpyHostToDevice);

	/**
	 * Declaracion de dimensiones de grid y tamaño de bloques
	 */
	dim3 dimGrid(width/dimOfBlock,height/dimOfBlock);//numero de tiles
   	dim3 dimBlock(dimOfBlock,dimOfBlock);//tamanio de los tiles

	#if __TIEMPO__KERNELS__
   		cudaEventRecord(e_start);
	#endif

	GradientCalc<<<dimGrid,dimBlock>>>(width, height, param.d_input,  stop, param.d_p, param.d_q, param.d_pq, param.d_R,param.d_max);

	#if __TIEMPO__KERNELS__
		cudaEventRecord(e_stop);
		cudaEventSynchronize(e_stop);
		milliseconds=0;
		cudaEventElapsedTime(&milliseconds, e_start, e_stop);
		t_kernel.gradiente+=milliseconds;
	#endif
	cudaMemcpy(max,param.d_max,1*sizeof(int),cudaMemcpyDeviceToHost);

	#if __TIEMPO__KERNELS__
		cudaEventRecord(e_start);
	#endif

	#if __HARRIS__ == 3
	harrisResponseFunction<<<dimGrid,dimBlock>>>(diff, width, height, param.d_R, param.d_p, param.d_q, param.d_pq, 1023,despl);
	#endif

	#if __TIEMPO__KERNELS__
		cudaEventRecord(e_stop);
		cudaEventSynchronize(e_stop);
		milliseconds=0;
		cudaEventElapsedTime(&milliseconds, e_start, e_stop);
	#endif

	//#ifdef __MAXIMO__CPU__
	cudaMemcpy(R,param.d_R+ (HARRIS_WINDOW_SIZE / 2) * (width + 1),(nPixels - (HARRIS_WINDOW_SIZE / 2) * (width + 1))*sizeof(int),cudaMemcpyDeviceToHost);
	//#endif
;

	int *data = pOutputImage;
	int *d_data=param.d_R;

	#if __TIEMPO__KERNELS__
		cudaEventRecord(e_start);
	#endif
	// determine maximum value

	#ifdef __MAXIMO__GPU__

	int h_odata[param.numBlocks];

	dim3 adimBlock(param.numThreads, 1);
    	dim3 adimGrid(param.numBlocks, 1);

	reduce<<< adimGrid, adimBlock, param.smemSize >>>(d_data, param.d_odata,(unsigned int)nPixels,param.numThreads);
	cudaMemcpy(h_odata,param.d_odata,param.numBlocks*sizeof(int),cudaMemcpyDeviceToHost);


	for (int i=0; i<param.numBlocks; i++)
            {
                max[1]=maximum=MAX(h_odata[i],maximum );

            }
	#endif

	#ifdef __MAXIMO__CPU__
	maximum = 0;
	for (int i = 0; i < nPixels; ++i)
	{
		if (data[i] > maximum)
			maximum = data[i];

	}
	#endif

	#if __TIEMPO__KERNELS__
		cudaEventRecord(e_stop);
		cudaEventSynchronize(e_stop);
		milliseconds=0;
		cudaEventElapsedTime(&milliseconds, e_start, e_stop);
	#endif

	max[0] = int(max[0] * 0.005f);// + 0.5f
	int *pCandidateOffsets = new int[nPixels];
	int *nCandidates = new int[1];
	nCandidates[0]=0;
	// only accept good pixels

	int __GOODPIXELS__ = 1;

	if(__GOODPIXELS__==0){
		for (int i = 0; i < nPixels; i++)
		{
			if (data[i] >= max[0])
				pCandidateOffsets[nCandidates[0]++] = i;
		}
	}

	else if(__GOODPIXELS__==1){

		cudaMemcpy(param.d_nCandidates,nCandidates,sizeof(int)*2,cudaMemcpyHostToDevice);
		#if __TIEMPO__KERNELS__
			cudaEventRecord(e_start);
		#endif
		//goodPixels_shared<<<dimGrid,dimBlock>>>(d_data,param.d_pCandidateOffsets, param.d_nCandidates,width,height,max[0]);
		//goodPixels_global<<<dimGrid,dimBlock>>>(d_data,param.d_pCandidateOffsets, d_aux,param.d_nCandidates,width,height,max[0]);
		goodPixels(param,width,height,max,1);
		#if __TIEMPO__KERNELS__
			cudaEventRecord(e_stop);
			cudaEventSynchronize(e_stop);
			milliseconds=0;
			cudaEventElapsedTime(&milliseconds, e_start, e_stop);
			t_kernel.goodPixels+=milliseconds;
		#endif
		cudaMemcpy(nCandidates,param.d_nCandidates,sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(pCandidateOffsets,param.d_pCandidateOffsets,sizeof(int)*nCandidates[0],cudaMemcpyDeviceToHost);


		}
	#if __TIEMPO__KERNELS__
		cudaEventRecord(e_start);
	#endif
	#if __SORT__ == CPU
	QuicksortInverse(pCandidateOffsets, data, 0, nCandidates[0] - 1);
	//QuicksortInverse(pCandidateOffsets2, data, 0, nCandidates2[0] - 1);
	/*sort(pCandidateOffsets);
	sort();*/
	#endif
	#if __TIEMPO__KERNELS__
		cudaEventRecord(e_stop);
		cudaEventSynchronize(e_stop);
		milliseconds=0;
		cudaEventElapsedTime(&milliseconds, e_start, e_stop);
		t_kernel.qSort+=milliseconds;
	#endif

	//printf(" %f\t",milliseconds);
	float fMinDistance = 5.0f;
	const int nMinDistance = int(fMinDistance );//+ 0.5f
	#if __TIEMPO__KERNELS__
		cudaEventRecord(e_start);
	#endif
	unsigned char image[nPixels];
	for (int i=0;i<nPixels;i++) image[i]=0;
	int nInterestPoints = 0;
	const int nMaxPoints=700;
	for (int i = 0; i < nCandidates[0] && nInterestPoints < nMaxPoints; i++)
	{
		const int offset = pCandidateOffsets[i];

		const int x = offset % width;
		const int y = offset / width;

		bool bTake = true;

		const int minx = x - nMinDistance < 0 ? 0 : x - nMinDistance;
		const int miny = y - nMinDistance < 0 ? 0 : y - nMinDistance;
		const int maxx = x + nMinDistance >= width ? width - 1 : x + nMinDistance;
		const int maxy = y + nMinDistance >= height ? height - 1 : y + nMinDistance;
		const int diff = width - (maxx - minx + 1);

		for (int l = miny, offset2 = miny * width + minx; l <= maxy; l++, offset2 += diff)
			for (int k = minx; k <= maxx; k++, offset2++)
				if (image[l * width + k]==1)
				{
					bTake = false;
					break;
				}

		if (bTake)
		{
			// store  point
			//cout << "guarda:" << x << "," << y << endl;
			puntosDeInteres[nInterestPoints].x = float(x);
			puntosDeInteres[nInterestPoints].y = float(y);
			nInterestPoints++;

			// mark location in grid for distance constraint check
			image[offset] = 1;
		}
	}
	#if __TIEMPO__KERNELS__
		cudaEventRecord(e_stop);
		cudaEventSynchronize(e_stop);
		milliseconds=0;
		cudaEventElapsedTime(&milliseconds, e_start, e_stop);
		t_kernel.noParalelo+=milliseconds;
	#endif

	//cudaFree(raw_pointer_cast(d_pCandidateOffsets.data()));
	free(pCandidateOffsets);
	free(nCandidates);
	return nInterestPoints;
}



mutex m_index;
void run(int hilo, char ** fileList, int &count, int &n_veces, int &index, int &contador){

	//int contador=0;
	float milliseconds=0, media=0;
	int ancho,alto;
	bool memoria_reservada=false;
	cudaEvent_t start, stop;

	vector<punto> puntosDeInteres(700);

	parametros param;
	bool salir=false;
	CImgDisplay main_disp;
	int j;
	int n_ejec=0;
	tiempos t_kernel;
	t_kernel.gradiente=0.0;
	t_kernel.goodPixels=0.0;
	t_kernel.noParalelo=0.0;
	t_kernel.total=0.0;
	t_kernel.qSort=0.0;
			// main loop
			//cout << "count: " << count << endl;
			while(contador<n_veces )
			{
				m_index.lock();
				if(index<count){
					j=index;
					index++;
				}
				else {
					j=0;
					index = 0;
					contador++;
				}
				m_index.unlock();

				unsigned char *grayImage = new unsigned char[640*480];

				/**
				 * Convierte la imagen fileList[index] a escala de grises y la guarda
				 * en grayimage
				 */
				ReadBMP(fileList[j], grayImage, ancho, alto);

				if(memoria_reservada==false){
					reserva_memoria(param,ancho,alto);
					memoria_reservada=true;
				}
				CImg<unsigned char> image(grayImage,ancho,alto);
				//main_disp.display(image);


				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				#if __TIEMPO__TOTAL__
					cudaEventRecord(start);
				#endif
				const int nPuntos=cornerDetector(grayImage,puntosDeInteres, ancho, alto, param,t_kernel);
				#if __TIEMPO__TOTAL__
					cudaEventRecord(stop);
					cudaEventSynchronize(stop);
					milliseconds=0;
					cudaEventElapsedTime(&milliseconds, start, stop);
					media=media+milliseconds;
					t_kernel.total+=milliseconds;
				#endif

				CImg<unsigned char> img(grayImage,ancho,alto);
				const unsigned char color[3] = {0,100,255};
				for(int i=0;i<nPuntos;i++){
					//img.draw_rectangle(x0,y0,x1,y1,color,1);
					//img.draw_point(puntosDeInteres[i].x,puntosDeInteres[i].y,color);
					img.draw_circle(puntosDeInteres[i].x,puntosDeInteres[i].y,2,color);

				}
				//CImg<unsigned char> image(grayImage,ancho,alto);
				main_disp.display(img);

				free(grayImage);
				n_ejec++;
			}

	main_disp.close();

	cout << "tiempo medio del hilo: " << media/n_ejec << endl;
	cout << "tiempo hilo*n_imagenes/imagenes totales: " << media/((contador)*count) << endl;
	cout << "n_ejec: " << n_ejec << endl;

	cout << "Media gradiente hilo ("<< hilo << "): " << (float)t_kernel.gradiente/n_ejec << endl;
	cout << "Media Good_Pixels hilo ("<< hilo << "): " <<  (float)t_kernel.goodPixels/n_ejec << endl;
	cout << "Media qSort hilo ("<< hilo << "): " <<  (float)t_kernel.qSort/n_ejec << endl;
	cout << "Media noParalelo hilo ("<< hilo << "): " <<  (float)t_kernel.noParalelo/n_ejec << endl;

	cout << "Media total hilo ("<< hilo << "): " <<  (float)t_kernel.total/n_ejec << endl;
	libera_memoria(param);
}



int main(void)

{

	char *fileList[250];
	int count = 0;
	int contador=0;
	int op=0;
	printf("Selecciona el video para procesar: \n");
	printf("\t(0) corridor\n");
	printf("\t(1) bunny\n");
	printf("\t(2) museum_wall\n");
	printf("\t(3) Robot_complex\n");
	printf("\t(4) Robot_simple\n");
	printf("\t(5) Brick_wall\n");
	scanf("%d",&op);

	char DEMO_IMAGE[250];
	char SRC_IMAGE[250];
	strcpy(DEMO_IMAGE,__DIRECTORIO__RAIZ__);
	strcpy(SRC_IMAGE,__DIRECTORIO__RAIZ__);
	switch (op){
		case 0:
			strcat(DEMO_IMAGE,"/corridor/0001.bmp");
			strcat(SRC_IMAGE,"/corridor/");
			break;
		case 1:
			strcat(DEMO_IMAGE,"/bunny/0001.bmp");
			strcat(SRC_IMAGE,"/bunny/");
			break;
		case 2:
			strcat(DEMO_IMAGE,"/museum_wall/0001.bmp");
			strcat(SRC_IMAGE,"/museum_wall/");
			break;
		case 3:
			strcat(DEMO_IMAGE,"/robot_complex/0001.bmp");
			strcat(SRC_IMAGE,"/robot_complex/");
			break;
		case 4:
			strcat(DEMO_IMAGE,"/robot_simple/0001.bmp");
			strcat(SRC_IMAGE,"/robot_simple/");
			break;
		case 5:
			strcat(DEMO_IMAGE,"/brick_wall/0001.bmp");
			strcat(SRC_IMAGE,"/brick_wall/");
			break;
	}
	cout << "eleccion demo: " << DEMO_IMAGE << endl;
	cout << "eleccion src: " << SRC_IMAGE << endl;

	GenFileListSorted(SRC_IMAGE, fileList, &count);
	unsigned int n = std::thread::hardware_concurrency();
	int n_veces;
	cout << "Introduce el numero de repeticiones: ";
	cin >> n_veces;
	cout << endl;

	cout << "Introduce el numero de hilos: ";
	cin >> n;
	cout << endl;

	thread *v_thread= new thread[n];
	int index=0;

	for(int t_id=0;t_id<n;t_id++){
		v_thread[t_id]=thread(run,t_id,fileList,std::ref(count),std::ref(n_veces),std::ref(index),std::ref(contador));
	}

	for(int t_id=0;t_id<n;t_id++){
		v_thread[t_id].join();
	}
	delete []v_thread;
}
