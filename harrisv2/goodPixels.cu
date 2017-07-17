/**
  @file goodPixels.cu
  @brief Implementación de la función GoodPixels

  En este fichero se encuentran las funciones que se encargan de localizar
  los puntos de la imagen que podrian ser considerados una esquina en esta.

  @author Julio González Carranza
  @date 10/03/2017

*/
#include "defines.hpp"


/**
 @brief Determina la forma en la que se procesaran los datos para calcular los pixeles buenos

 Esta funcion determinara de que forma se procesaran los datos para calcular los pixeles buenos.
 Existen varias formas para realizar estos calculos, como pueden ser utilizando memoria global,
 texturas o memoria compartida.

 @param param Estructura del hilo con los datos necesarios para realizar la computacion
 @param width Alto de la imagen
 @param height Ancho de la imagen
 @param d_max Valor máximo obtenido por la funcion Harris
 @param op Valor que representa la opcion elegida para realizar el procesamiento

 */
void goodPixels(parametros &param, int width,int height,int *d_max, int op){
	dim3 dimGrid(width/dimOfBlock,height/dimOfBlock);//numero de tiles
	dim3 dimBlock(dimOfBlock,dimOfBlock);//tamanio de los tiles
	switch(op){
		case 0:
			goodPixels_shared<<<dimGrid,dimBlock>>>(param.d_R,param.d_pCandidateOffsets, param.d_nCandidates,width,height,d_max[0]);
			break;
		case 1:
			goodPixels_global<<<dimGrid,dimBlock>>>(param.d_R,param.d_pCandidateOffsets, param.d_nCandidates,width,height,d_max[0]);
			break;
		case 2:



			//cudaArray *array;
			//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindUnsigned);
			//cudaMallocArray(&array,&channelDesc,height,width);
			//cudaMemcpyToArray(array,0,0,d_data,width*height*sizeof(int),cudaMemcpyDeviceToDevice);
			// Specify texture
			//cudaResourceDesc resDesc;
			//memset(&resDesc, 0, sizeof(resDesc));
			//resDesc.resType = cudaResourceTypeLinear;
			//resDesc.res.linear.devPtr = d_data;
			//resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
  		//resDesc.res.linear.desc.x = 32; // bits per channel
  		//resDesc.res.linear.sizeInBytes = width*height*sizeof(float);

		    // Specify texture object parameters
			//cudaTextureDesc texDesc;
			//memset(&texDesc, 0, sizeof(texDesc));
			//texDesc.readMode = cudaReadModeElementType;
			//texDesc.addressMode[0]   = cudaAddressModeWrap;
			//texDesc.addressMode[1]   = cudaAddressModeWrap;
			//texDesc.filterMode       = cudaFilterModeLinear;
			//texDesc.readMode         = cudaReadModeElementType;
			//texDesc.normalizedCoords = 1;

			// Create texture object
			//cudaTextureObject_t texture_array = 0;
			//cudaCreateTextureObject(&texture_array, &resDesc, &texDesc, NULL);
			//std::cout << "texture" << std::endl;
			goodPixels_texture<<<dimGrid,dimBlock>>>(param.texture_array,param.d_pCandidateOffsets, param.d_nCandidates,width,height,d_max[0]);

			//cudaDestroyTextureObject(texture_array);
			break;
	}
}


/**
 @brief Localiza los posibles pixeles candidatos a ser esquina en la imagen

 Para determinar si un pixel es una esquina, es necesario consultar el valor
 que obtuvieron sus vecinos en la función de harris.
 Si un pixel dado es mayor que sus vecinos, es posible que este sea una esquina

 Implementación consultando vecinos y utilizando memoria global

 @param d_data Valores de harris de la imagen, en memoria de dispositivo
 @param d_pCandidateOffsets Vector en dispositivo en el que se guardaran las posiciones de los candidatos
 @param d_nCandidateOffsets Vector en dispositivo, en la posición [0] se guardará el número de candidatos de la imagen
 @param width Alto de la imagen
 @param height Ancho de la imagen
 @param d_max Valor máximo obtenido por la funcion Harris

 */

__global__ void goodPixels_global(int *d_data, int *d_pCandidateOffsets, int *d_nCandidates, int width,int height,int d_max){

	int row= blockIdx.y*dimOfBlock+threadIdx.y;
	int col= blockIdx.x*dimOfBlock+threadIdx.x;
	int id=row*width+col;

	/*__shared__ float s_candidates[dimOfBlock][dimOfBlock];
	s_candidates[threadIdx.y][threadIdx.x]=d_data[id];
	float val=s_candidates[threadIdx.y][threadIdx.x];*/
	float val = d_data[id];
	//__syncthreads();

	if(col > 0 && col < width-1 && row > 0 && row < height-1 && d_data[id]>=d_max){

		//superiores
		val = fmaxf(val,d_data[id-width-1]);
		val = fmaxf(val,d_data[id-width]);
		val = fmaxf(val,d_data[id-width+1]);

		val = fmaxf(val,d_data[id-1]);
		val = fmaxf(val,d_data[id+1]);

		val = fmaxf(val,d_data[id+width-1]);
		val = fmaxf(val,d_data[id+width]);
		val = fmaxf(val,d_data[id+width+1]);

		if(d_data[id] == val){ //el era el mayor
			float temp = atomicAdd(&d_nCandidates[0],1);
			//d_aux[temp]=d_data[id];
			d_pCandidateOffsets[(int)temp]=id;
		}
    }
    return;
}


/**
 @brief Localiza los posibles pixeles candidatos a ser esquina en la imagen

 Para determinar si un pixel es una esquina, es necesario consultar el valor
 que obtuvieron sus vecinos en la función de harris.
 Si un pixel dado es mayor que sus vecinos, es posible que este sea una esquina

 Implementación consultando vecinos y utilizando memoria compartida

 @param d_data Valores de harris de la imagen, en memoria de dispositivo
 @param d_pCandidateOffsets Vector en dispositivo en el que se guardaran las posiciones de los candidatos
 @param d_nCandidateOffsets Vector en dispositivo, en la posición [0] se guardará el número de candidatos de la imagen
 @param width Alto de la imagen
 @param height Ancho de la imagen
 @param d_max Valor máximo obtenido por la funcion Harris

 */

__global__ void goodPixels_shared(int *d_data, int *d_pCandidateOffsets, int *d_nCandidates, int width,int height,int d_max){


    int row= blockIdx.y*dimOfBlock+threadIdx.y;
	int col= blockIdx.x*dimOfBlock+threadIdx.x;
	int id=row*width+col;
	int idx, idy;
	idx=threadIdx.x+1;
	idy=threadIdx.y+1;
	__shared__ float s_candidates[dimOfBlock+3][dimOfBlock+3];
	//__shared__ float s_val[dimOfBlock+1][dimOfBlock+1];
	//s_candidates[idy][idx]=d_data[id
	s_candidates[idy][idx]=s_candidates[threadIdx.y][threadIdx.x];

	float val=d_data[id];
	float orig_val=d_data[id];
	// superiores
if(col > 0 && col < width-1 && row > 0 && row < height-1){
		if(threadIdx.y==0){
			if(threadIdx.x==0){
				//superior izquierdo
				s_candidates[idy-1][idx-1]=d_data[id-width-1];
				//laterales izquierdo
				s_candidates[idy][idx-1]=d_data[id-1];
			}
			else if(threadIdx.x==dimOfBlock-1){
				//superior derecho
				s_candidates[idy-1][idx+1]=d_data[id-width+1];
				//laterales derecho
				s_candidates[idy][idx+1]=d_data[id+1];
			}
			//superiores
			s_candidates[idy-1][idx]=d_data[id-width];
		}
		//laterales
		if(threadIdx.x==0)s_candidates[idy][idx-1]=d_data[id-1];
		if(threadIdx.x==dimOfBlock-1)s_candidates[idy][idx+1]=d_data[id+1];

		//inferiores
		if(threadIdx.y==dimOfBlock-1){
			if(threadIdx.x==0){
				//inferiores izquierdo
				s_candidates[idy+1][idx-1]=d_data[id+width-1];
			}
			else if(threadIdx.x==dimOfBlock-1){
				//inferiores derecho
				s_candidates[idy+1][idx+1]=d_data[id+width+1];
			}
			//inferiores
			s_candidates[idy+1][idx]=d_data[id+width];
		}
		//centrales
		//s_candidates[idy][idx]=d_data[id];
		//s_val[idy-1][idx-1]=s_candidates[idy][idx];
	}
	else return;

	__syncthreads();
	if(val<=d_max){
	//&& s_val[idy-1][idx-1]>=d_max


		val = MAX(val,s_candidates[idy-1][idx-1]);
		val = MAX(val,s_candidates[idy-1][idx]);
		val = MAX(val,s_candidates[idy-1][idx+1]);

		val = MAX(val,s_candidates[idy][idx-1]);
		val = MAX(val,s_candidates[idy][idx+1]);

		val = MAX(val,s_candidates[idy+1][idx-1]);
		val = MAX(val,s_candidates[idy+1][idx]);
		val = MAX(val,s_candidates[idy+1][idx+1]);

		/*if(val<s_candidates[idy-1][idx-1]) return;
		if(val<s_candidates[idy-1][idx]) return;
		if(val<s_candidates[idy-1][idx+1]) return;

		if(val<s_candidates[idy][idx-1]) return;
		if(val<s_candidates[idy][idx+1]) return;

		if(val<s_candidates[idy+1][idx-1]) return;
		if(val<s_candidates[idy+1][idx]) return;
		if(val<s_candidates[idy+1][idx+1]) return;*/



		//if(s_candidates[idy][idx] == s_val[idy-1][idx-1]){ //el era el mayor
		if(val<=orig_val){
			int temp = atomicAdd(&d_nCandidates[0],1);
			d_pCandidateOffsets[temp]=id;
		}
    }
    else return;

}

/**
 @brief Localiza los posibles pixeles candidatos a ser esquina en la imagen

 Para determinar si un pixel es una esquina, es necesario consultar el valor
 que obtuvieron sus vecinos en la función de harris.
 Si un pixel dado es mayor que sus vecinos, es posible que este sea una esquina

 Implementación consultando vecinos y utilizando memoria de texturas

 @param d_data Valores de harris de la imagen, en memoria como texturas
 @param d_pCandidateOffsets Vector en dispositivo en el que se guardaran las posiciones de los candidatos
 @param d_nCandidateOffsets Vector en dispositivo, en la posición [0] se guardará el número de candidatos de la imagen
 @param width Alto de la imagen
 @param height Ancho de la imagen
 @param d_max Valor máximo obtenido por la funcion Harris

 */

__global__ void goodPixels_texture(cudaTextureObject_t d_data, int *d_pCandidateOffsets, int *d_nCandidates, int width, int height, int d_max){

	int row= blockIdx.y*dimOfBlock+threadIdx.y;
	int col= blockIdx.x*dimOfBlock+threadIdx.x;
	int id=row*width+col;

	/*__shared__ float s_candidates[dimOfBlock][dimOfBlock];
	s_candidates[threadIdx.y][threadIdx.x]=d_data[id];
	float val=s_candidates[threadIdx.y][threadIdx.x];*/
	int val = tex2D<int>(d_data,col,row);
	//int val = tex1Dfetch<int>(d_data,id);
	int t_val=val;

	if(col > 0 && col < width-1 && row > 0 && row < height-1 && val>=d_max){

		//superiores
		val = MAX(val,tex2D<int>(d_data,col-1,row-1));
		val = MAX(val,tex2D<int>(d_data,col,row-1));
		val = MAX(val,tex2D<int>(d_data,col+1,row-1));

		val = MAX(val,tex2D<int>(d_data,col-1,row));
		val = MAX(val,tex2D<int>(d_data,col+1,row));

		val = MAX(val,tex2D<int>(d_data,col-1,row+1));
		val = MAX(val,tex2D<int>(d_data,col,row+1));
		val = MAX(val,tex2D<int>(d_data,col+1,row+1));


/*		val = MAX(val,tex1Dfetch<int>(d_data,id-width-1));
		val = MAX(val,tex1Dfetch<int>(d_data,id-width));
		val = MAX(val,tex1Dfetch<int>(d_data,id-width+1));

		val = MAX(val,tex1Dfetch<int>(d_data,id-1));
		val = MAX(val,tex1Dfetch<int>(d_data,id+1));

		val = MAX(val,tex1Dfetch<int>(d_data,id+width-1));
		val = MAX(val,tex1Dfetch<int>(d_data,id+width));
		val = MAX(val,tex1Dfetch<int>(d_data,id+width+1));
*/


		//if(tex2D<int>(d_data,col,row) == val){ //el era el mayor
		if(t_val == val){ //el era el mayor
			int temp = atomicAdd(&d_nCandidates[0],1);
			//d_aux[temp]=d_data[id];
			d_pCandidateOffsets[(int)temp]=id;
		}
    }
}
