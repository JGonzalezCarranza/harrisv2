/*
 ============================================================================
 Name        : harris_v1.0.cu
 Author      : Julio Gonzalez Carranza
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
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

#define DEMO_IMAGE	"/home/julio/universidad/proyecto/proyecto/HarrisCornerRAwareEval/corridor/0001.bmp"
#define SRC_IMAGE		"/home/julio/universidad/proyecto/proyecto/HarrisCornerRAwareEval/corridor/"

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
/*    int i;
    FILE* f = fopen(filename, "rb");

    if(f == NULL)
        throw "Argument Exception";

    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];
*/
    /*cout << endl;
    cout << "  Name: " << filename << endl;
    cout << " Width: " << width << endl;
    cout << "Height: " << height << endl;*/
/*    ancho = width;
    alto = height;
    int row_padded = (width*3 + 3) & (~3);
    unsigned char* data = new unsigned char[row_padded];

    unsigned char tmp;

    for(int i = 0; i < height; i++)
    {
        fread(data, sizeof(unsigned char), row_padded, f);
        for(int j = 0; j < width*3; j += 3)
        {
            // Convert (B, G, R) to (R, G, B)
            tmp = data[j];
            data[j] = data[j+2];
            data[j+2] = tmp;
            //cout << "[" << i*width+j/3 << "]= " << (data[j]+data[j+1]+data[j+2])/3 << endl;

            //funciona
            //grayData[i*width+j/3] = (int)(data[j]+data[j+1]+data[j+2])/3;
            //prueba
            grayData[i*width+j/3]=(data[j]+(data[j+1]<<1)+data[j+2]+2)>>2;
            //funciona
            //grayData[i*width+j/3] = int(data[j]*0.299+data[j+1]*0.587+data[j+2]*0.114);

            //cout << "[" << i*width+j/3 << "]" << endl;
            //cout << "R: "<< (int)data[j] << " G: " << (int)data[j+1]<< " B: " << (int)data[j+2]<< endl;
            //cout << "data["<< i << "," << j/3 << "]= " << (unsigned int)grayData[i*width+j/3] << endl;
            //printf("data: %d\n",grayData[i*width+j/3]);
        }
    }
    free(data);

    fclose(f);
*/
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

int cornerDetector(unsigned char* grayImage, vector<punto> &puntosDeInteres, const int ancho, const int alto, parametros &param, tiempos t_kernel){

	const unsigned char *input = grayImage;
	unsigned int *pixHist;
	const int nPixels = ancho*alto;
	const int width = ancho;
	const int height = alto;
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
		printf("calculo gradiente: %f\n",milliseconds);
	#endif
	cudaMemcpy(max,param.d_max,1*sizeof(int),cudaMemcpyDeviceToHost);

	free(pixHist);


	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////

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
	printf("calculo harris: %f\n",milliseconds);
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
		printf("maximo: %f\n",milliseconds);
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
		int *d_aux;

		cudaMemcpy(param.d_nCandidates,nCandidates,sizeof(int)*2,cudaMemcpyHostToDevice);
		#if __TIEMPO__KERNELS__
		cudaEventRecord(e_start);
		#endif
		goodPixels<<<dimGrid,dimBlock>>>(d_data,param.d_pCandidateOffsets, d_aux,param.d_nCandidates,width,height,max[0]);
		#if __TIEMPO__KERNELS__
		cudaEventRecord(e_stop);
		cudaEventSynchronize(e_stop);
		milliseconds=0;
		cudaEventElapsedTime(&milliseconds, e_start, e_stop);
		t_kernel.goodPixels+=milliseconds;
		printf("good pixels: %f\n",milliseconds);
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
		printf("quicksort: %f\n",milliseconds);
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
		printf("puntos finales: %f\n",milliseconds);
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
					printf("TIEMPO TOTAL: %f\n",milliseconds);
					media=media+milliseconds;
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
	libera_memoria(param);
}



int main(void)

{

	char *fileList[250];
	int count = 0;
	int contador=0;
	GenFileListSorted(SRC_IMAGE, fileList, &count);
	unsigned int n = std::thread::hardware_concurrency();
	n=1;
	thread *v_thread= new thread[n];
	int n_veces=5;
	int index=0;
	for(int t_id=0;t_id<n;t_id++){
		v_thread[t_id]=thread(run,t_id,fileList,std::ref(count),std::ref(n_veces),std::ref(index),std::ref(contador));
	}

	for(int t_id=0;t_id<n;t_id++){
		v_thread[t_id].join();
	}
	delete []v_thread;
}


