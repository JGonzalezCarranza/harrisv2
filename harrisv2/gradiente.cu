#include "defines.hpp"


__global__ void GradientCalc(int width,int height, const unsigned char *input,
		 const int stop, int *d_p, int *d_q, int *d_pq, int *R, int *max){//se le debe pasar el offset*

	int row= blockIdx.y*dimOfBlock+threadIdx.y;
	int col= blockIdx.x*dimOfBlock+threadIdx.x;
	int id=row*width+col;
	//int temp=0;
	
	R[id]=0;
	int *d_R;
	d_R=R;
	
	int p, q;
	int ppsum, pqsum, qqsum;
	ppsum=0;
	pqsum=0;
	qqsum=0;
	int offset;
	offset=0;
	//if(id>=width+1 && id<stop){
	if(row >= 2 && row < height-2 && col >= 2 && col < width-2 ){
		
		//primeros 3
		p=(int)(input[id+1+offset]-input[id-1+offset]);
		q=(int)(input[id+width+offset]-input[id-width+offset]);
		ppsum=ppsum+p*p;
		qqsum=qqsum+q*q;
		pqsum=pqsum+p*q;
		
		offset++;
		
		p=(int)(input[id+1+offset]-input[id-1+offset]);
		q=(int)(input[id+width+offset]-input[id-width+offset]);
		ppsum=ppsum+p*p;
		qqsum=qqsum+q*q;
		pqsum=pqsum+p*q;
		
		offset++;
		
		p=(int)(input[id+1+offset]-input[id-1+offset]);
		q=(int)(input[id+width+offset]-input[id-width+offset]);
		ppsum=ppsum+p*p;
		qqsum=qqsum+q*q;
		pqsum=pqsum+p*q;
		
		/**
		*
		*
		*/
		offset=offset-2;
		offset=offset+width;
		
		p=(int)(input[id+1+offset]-input[id-1+offset]);
		q=(int)(input[id+width+offset]-input[id-width+offset]);
		ppsum=ppsum+p*p;
		qqsum=qqsum+q*q;
		pqsum=pqsum+p*q;
		
		offset++;
		
		p=(int)(input[id+1+offset]-input[id-1+offset]);
		q=(int)(input[id+width+offset]-input[id-width+offset]);
		ppsum=ppsum+p*p;
		qqsum=qqsum+q*q;
		pqsum=pqsum+p*q;
		
		offset++;
		
		p=(int)(input[id+1+offset]-input[id-1+offset]);
		q=(int)(input[id+width+offset]-input[id-width+offset]);
		ppsum=ppsum+p*p;
		qqsum=qqsum+q*q;
		pqsum=pqsum+p*q;
		/**
		*
		*
		*/
		offset=offset-2;
		offset=offset+width;
		
		p=(int)(input[id+1+offset]-input[id-1+offset]);
		q=(int)(input[id+width+offset]-input[id-width+offset]);
		ppsum=ppsum+p*p;
		qqsum=qqsum+q*q;
		pqsum=pqsum+p*q;
		
		offset++;
		
		p=(int)(input[id+1+offset]-input[id-1+offset]);
		q=(int)(input[id+width+offset]-input[id-width+offset]);
		ppsum=ppsum+p*p;
		qqsum=qqsum+q*q;
		pqsum=pqsum+p*q;
		
		offset++;
		
		p=(int)(input[id+1+offset]-input[id-1+offset]);
		q=(int)(input[id+width+offset]-input[id-width+offset]);
		ppsum=ppsum+p*p;
		qqsum=qqsum+q*q;
		pqsum=pqsum+p*q;
		
		d_R[id] =(float) (ppsum * qqsum - pqsum * pqsum) - (float)(0.04*(ppsum + qqsum) * (ppsum + qqsum));
		atomicMax(&max[0],d_R[id]);
	}
	
}
