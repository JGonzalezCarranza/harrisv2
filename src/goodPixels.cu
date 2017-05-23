#include "defines.hpp"

__global__ void goodPixels(int *d_data, int *d_pCandidateOffsets, int *d_aux, int *d_nCandidates, int width,int height,int d_max){

	int row= blockIdx.y*dimOfBlock+threadIdx.y;
	int col= blockIdx.x*dimOfBlock+threadIdx.x;
	int id=row*width+col;

	/*__shared__ float s_candidates[dimOfBlock][dimOfBlock];
	s_candidates[threadIdx.y][threadIdx.x]=d_data[id];    
	float val=s_candidates[threadIdx.y][threadIdx.x];*/
	float val = d_data[id];
	__syncthreads();

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
    
    /*
    int row= blockIdx.y*dimOfBlock+threadIdx.y;
	int col= blockIdx.x*dimOfBlock+threadIdx.x;
	int id=row*width+col;
	int idx, idy;
	idx=threadIdx.x+1;
	idy=threadIdx.y+1;
	__shared__ float s_candidates[dimOfBlock+3][dimOfBlock+3];
	//__shared__ float s_val[dimOfBlock+1][dimOfBlock+1];
	s_candidates[idy][idx]=d_data[id]; 
	   
	float val=s_candidates[threadIdx.y][threadIdx.x];
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
		//float val = s_candidates[threadIdx.y][threadIdx.x];
	}
	*/
/*	__syncthreads();
	if(col > 0 && col < width-1 && row > 0 && row < height-1 && s_candidates[idy][idx] >=d_max){//&& s_val[idy-1][idx-1]>=d_max

*/
		/*s_val[idy-1][idx-1] = fmaxf(s_val[idy-1][idx-1],s_candidates[idy-1][idx-1]);
		s_val[idy-1][idx-1] = fmaxf(s_val[idy-1][idx-1],s_candidates[idy-1][idx]);
		s_val[idy-1][idx-1] = fmaxf(s_val[idy-1][idx-1],s_candidates[idy-1][idx+1]);
	
		s_val[idy-1][idx-1] = fmaxf(s_val[idy-1][idx-1],s_candidates[idy][idx-1]);
		s_val[idy-1][idx-1] = fmaxf(s_val[idy-1][idx-1],s_candidates[idy][idx+1]);
		
		s_val[idy-1][idx-1] = fmaxf(s_val[idy-1][idx-1],s_candidates[idy+1][idx-1]);
		s_val[idy-1][idx-1] = fmaxf(s_val[idy-1][idx-1],s_candidates[idy+1][idx]);
		s_val[idy-1][idx-1] = fmaxf(s_val[idy-1][idx-1],s_candidates[idy+1][idx+1]);*/
		
/*		if(s_candidates[idy][idx]<s_candidates[idy-1][idx-1]) return;
		if(s_candidates[idy][idx]<s_candidates[idy-1][idx]) return;
		if(s_candidates[idy][idx]<s_candidates[idy-1][idx+1]) return;
		
		if(s_candidates[idy][idx]<s_candidates[idy][idx-1]) return;
		if(s_candidates[idy][idx]<s_candidates[idy][idx+1]) return;
		
		if(s_candidates[idy][idx]<s_candidates[idy+1][idx-1]) return;
		if(s_candidates[idy][idx]<s_candidates[idy+1][idx]) return;
		if(s_candidates[idy][idx]<s_candidates[idy+1][idx+1]) return;
		
		
		
		//if(s_candidates[idy][idx] == s_val[idy-1][idx-1]){ //el era el mayor
		if(s_candidates[idy][idx] >=d_max){
			int temp = atomicAdd(&d_nCandidates[0],1);		
			//d_aux[temp]=d_data[id];
			d_pCandidateOffsets[temp]=id;
		}
    }
    else return;  */
}
