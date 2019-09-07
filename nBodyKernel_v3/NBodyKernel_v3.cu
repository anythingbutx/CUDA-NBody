
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//gtx 1060: N = 20480
//kernel execution time = 14.6423368454 milliseconds




#define BLOCK 1024 
#define N 20480 

#define DAMP 0.5f
#define EPSILON 0.000001f
#define DT 0.001f
#define G 1.0f
#define H 1.0f


// Globals
float4 p[N], *p_GPU;
float3 v[N], *v_GPU;

__global__ void getVelocity(float4 *pos, float3 *vel) {
	
	const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float4 sharedPos[BLOCK];
	const float4 myPos = pos[id];
	float4 force = { 0.0f, 0.0f, 0.0f };

	for (int gMem_Index = threadIdx.x; gMem_Index < N; gMem_Index += blockDim.x) {

		sharedPos[threadIdx.x] = pos[gMem_Index]; __syncthreads();
		
		for (int i = 0; i < blockDim.x; i++) {
			float dx = sharedPos[i].x - myPos.x;
			float dy = sharedPos[i].y - myPos.y;
			float dz = sharedPos[i].z - myPos.z;

			float r2 = dx*dx + dy*dy + dz*dz + EPSILON;
			float r = 1.0f / sqrtf(r2);
			float mag = (G*myPos.w*sharedPos[i].w) / (r2)-(H*myPos.w*sharedPos[i].w) / (r2*r2);

			force.x += mag * dx * r;
			force.y += mag * dy * r;
			force.z += mag * dz * r;
		}
	}
	vel[id].x += ((force.x - DAMP*vel[id].x) / myPos.w)*DT;
	vel[id].y += ((force.y - DAMP*vel[id].y) / myPos.w)*DT;
	vel[id].z += ((force.z - DAMP*vel[id].z) / myPos.w)*DT;
}

__global__ void move(float4 *pos, float3 *vel) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= N) return;
	pos[id].x += vel[id].x*DT;
	pos[id].y += vel[id].y*DT;
	pos[id].z += vel[id].z*DT;
}




void set_initail_conditions()
{
	int i, j, k, num, particles_per_side;
	float position_start, temp, initail_seperation;

	temp = pow((float)N, 1.0 / 3.0) + 0.99999;
	particles_per_side = temp;
	//printf("\n cube root of N = %d \n", particles_per_side);
	position_start = -(particles_per_side - 1.0) / 2.0;
	initail_seperation = 2.0;

	for (i = 0; i<N; i++)
	{
		p[i].w = 1.0f;
	}

	num = 0;
	for (i = 0; i<particles_per_side; i++)
	{
		for (j = 0; j<particles_per_side; j++)
		{
			for (k = 0; k<particles_per_side; k++)
			{
				if (N <= num) break;
				p[num].x = position_start + i*initail_seperation;
				p[num].y = position_start + j*initail_seperation;
				p[num].z = position_start + k*initail_seperation;
				v[num].x = 0.0;
				v[num].y = 0.0;
				v[num].z = 0.0;
				num++;
			}
		}
	}
}

void cuda_NBody(int runs) {

	dim3 block, grid;
	block.x = BLOCK;
	block.y = 1; block.z = 1;
	grid.x = (N - 1) / block.x + 1;
	grid.y = 1; grid.z = 1;

	cudaMalloc((void**)&p_GPU, N * sizeof(float4));
	cudaMalloc((void**)&v_GPU, N * sizeof(float3));

	cudaMemcpy(p_GPU, p, N * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(v_GPU, v, N * sizeof(float3), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (float i = 0; i < runs; i++) {
		getVelocity <<<grid, block >>>(p_GPU, v_GPU);
		move <<<grid, block >>>(p_GPU, v_GPU);
	}
	cudaEventRecord(stop);
	cudaMemcpy(p, p_GPU, N * sizeof(float4), cudaMemcpyDeviceToHost);

	cudaError_t cudaEX = cudaGetLastError();
	if (cudaEX != cudaSuccess) {
		printf("cudaError: %s\n", cudaGetErrorString(cudaEX));
	}
	cudaEventSynchronize(stop); float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n\nGPU time = %0.10f milliseconds\n", elapsedTime / runs);
	cudaFree(p_GPU); cudaFree(v_GPU);
	cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
	cudaSetDevice(0);
	set_initail_conditions();
	cuda_NBody(1000);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}
