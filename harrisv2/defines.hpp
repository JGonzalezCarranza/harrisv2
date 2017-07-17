/*
 * defines.hpp
 *
 *  Created on: 6/2/2017
 *      Author: julio
 */

#ifndef DEFINES_HPP_
#define DEFINES_HPP_

#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <cuda.h>
#include <dirent.h>
#include <vector>

#define HARRIS_WINDOW_SIZE	3
#define dimOfBlock 16


typedef struct{
	int x;
	int y;
}punto;
typedef struct{
	float gradiente;
	float harris;
	float goodPixels;
	float qSort;
	float noParalelo;
	float total;
}tiempos;
typedef struct{
	int *d_p;
	int *d_q;
	int *d_pq;
	cudaResourceDesc resDesc;
	cudaTextureDesc texDesc;
	cudaTextureObject_t texture_array;
	//unsigned int *d_pixHist;
	int *d_max;
	unsigned char *d_input;
	int *d_R;
	int *d_nCandidates;
	int *d_pCandidateOffsets;
	int *d_odata;
	int numBlocks;
	int numThreads;
	int smemSize;
	//unsigned char *d_pCovImage;
}parametros;

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#ifndef MAX
#define MAX(x,y) ((x > y) ? x : y)
#endif

__global__ void GradientCalc(int width,int height, const unsigned char *input,
		 const int stop, int *d_p, int *d_q, int *d_pq, int *R, int *max);

__global__ void harrisResponseFunction(int diff, int width, int height, int *R,int *p,int *q, int *pq,int maskThreshold,  int despl);

void QuicksortInverse(int *pOffsets, const int *pValues, int nLow, int nHigh);

void goodPixels(parametros &param, int width,int height,int *d_max, int op);
__global__ void goodPixels_global(int *d_data, int *d_pCandidateOffsets, int *d_nCandidates, int width,int height,int d_max);
__global__ void goodPixels_shared(int *d_data, int *d_pCandidateOffsets, int *d_nCandidates, int width,int height,int d_max);
__global__ void goodPixels_texture(cudaTextureObject_t d_data, int *d_pCandidateOffsets, int *d_nCandidates, int width, int height, int d_max);

__device__ void warpReduce(volatile int *sdata, unsigned int tid, int blockSize);
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n, int blockSize);
unsigned int nextPow2(unsigned int x);
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);
void reserva_memoria(parametros &param, int ancho, int alto);
void libera_memoria(parametros &param);
#endif /* DEFINES_HPP_ */
