#include "defines.hpp"
void reserva_memoria(parametros &param, int ancho, int alto){
	printf("va a reservar\n");
	//param = (parametros *) malloc(sizeof(parametros));

	param.numBlocks=0;
	param.numThreads=0;

	int nPixels = ancho*alto;
	getNumBlocksAndThreads(6, nPixels, 32, 64, param.numBlocks, param.numThreads);
	param.smemSize = (param.numThreads <= 32) ? 2 * param.numThreads * sizeof(int) : param.numThreads *sizeof(int);

	cudaMalloc(&param.d_odata,param.numBlocks*sizeof(int));
	cudaMalloc(&param.d_p, nPixels*sizeof(int));
	cudaMalloc(&param.d_q, nPixels*sizeof(int));
	cudaMalloc(&param.d_pq, nPixels*sizeof(int));
	//cudaMalloc(&param.d_pixHist, 1024*sizeof(unsigned int));
	cudaMalloc(&param.d_nCandidates,sizeof(int)*2);
	cudaMalloc(&param.d_max, 2*sizeof(int));
	cudaMalloc(&param.d_input, nPixels*sizeof(unsigned char));
	cudaMalloc(&param.d_R, nPixels*sizeof(int));
	cudaMalloc(&param.d_pCandidateOffsets,nPixels*sizeof(int));
	//cudaMalloc(&param.d_pCovImage, nPixels*sizeof(unsigned char));
	printf("termina de reservar\n");
	return;
}

void libera_memoria(parametros &param){
	cudaFree(param.d_odata);
	cudaFree(param.d_p);
	cudaFree(param.d_q);
	cudaFree(param.d_pq);
	cudaFree(param.d_nCandidates);
	cudaFree(param.d_max);
	cudaFree(param.d_input);
	cudaFree(param.d_R);
	cudaFree(param.d_pCandidateOffsets);
}
