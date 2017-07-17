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


	/*memset(&param.resDesc, 0, sizeof(param.resDesc));
	param.resDesc.resType = cudaResourceTypeLinear;
	param.resDesc.res.linear.devPtr = param.d_R;
	param.resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
	param.resDesc.res.linear.desc.x = 32; // bits per channel
	param.resDesc.res.linear.sizeInBytes = ancho*alto*sizeof(int);*/

	memset(&param.resDesc, 0, sizeof(param.resDesc));
	param.resDesc.resType = cudaResourceTypePitch2D;
	param.resDesc.res.pitch2D.devPtr = param.d_R;
	param.resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
	param.resDesc.res.pitch2D.desc.x = 32; // bits per channel
	param.resDesc.res.pitch2D.width = ancho;
	param.resDesc.res.pitch2D.height = alto;
	param.resDesc.res.pitch2D.pitchInBytes = ancho*alto*sizeof(int);

	memset(&param.texDesc, 0, sizeof(param.texDesc));
	param.texDesc.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&param.texture_array, &param.resDesc, &param.texDesc, NULL);

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
	cudaDestroyTextureObject(param.texture_array);
}
