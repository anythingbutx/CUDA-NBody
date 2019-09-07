
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
//gtx 1060: N = 20480, BLOCK = 1024
//average time = 14.6543064117 milliseconds
//gtx 1060: N = 20470, BLOCK = 1024
//average time = 15.8831892014 milliseconds


#define BLOCK 1024
#define N 20470 //((1024 * 2) * 10) - 10

#define DAMP 0.5f
#define EPSILON 0.000001f
#define DT 0.001f
#define G 1.0f
#define H 1.0f

// Globals
float4 p[N];
float3 v[N];
float4 *p_GPU;
float3 *v_GPU;

__global__ void getVelocity(float4 *pos, float3 *vel) {
	const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float4 sharedPos[BLOCK];
	const float4 myPos = pos[id];
	float3 force = { 0.0f, 0.0f, 0.0f };

	for (int gMem_Index = threadIdx.x; true; gMem_Index += blockDim.x) {

		if (gMem_Index < N) sharedPos[threadIdx.x] = pos[gMem_Index];
		int len = __syncthreads_count(gMem_Index < N);

		if (len == 0) break;
		for (int i = 0; i < len; i++) {
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

__global__ void getVelocity_Remainder(float4 *pos, float3 *vel, int offset) {
	const unsigned int id = (threadIdx.x + blockDim.x * blockIdx.x) + offset;
	const float4 myPos = pos[id];
	float4 force = { 0.0f, 0.0f, 0.0f };

	for (int i = 0; i < N; i++) {
		float4 p = pos[i]; 

		float dx = p.x - myPos.x;
		float dy = p.y - myPos.y;
		float dz = p.z - myPos.z;

		float r2 = dx*dx + dy*dy + dz*dz + EPSILON;
		float r = 1.0f / sqrtf(r2);
		float mag = (G*p.w*myPos.w) / (r2)-(H*p.w*myPos.w) / (r2*r2);

		force.x += mag * dx * r;
		force.y += mag * dy * r;
		force.z += mag * dz * r;
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
	cudaMalloc((void**)&p_GPU, N * sizeof(float4));
	cudaMalloc((void**)&v_GPU, N * sizeof(float3));
	cudaMemcpy(p_GPU, p, N * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(v_GPU, v, N * sizeof(float3), cudaMemcpyHostToDevice);
}

void cuda_NBody(int runs) {

	cudaSetDevice(0);
	int fullGrid = (N - 1) / BLOCK + 1;
	int grid1 = N / BLOCK;
	int offset = grid1 * BLOCK;
	int remBLOCK = N - offset;

	cudaEvent_t start; cudaEventCreate(&start);
	cudaEvent_t stop; cudaEventCreate(&stop);
	
	if (grid1 < fullGrid) {

		cudaStream_t stream1; cudaStreamCreate(&stream1);
		cudaStream_t stream2; cudaStreamCreate(&stream2);

		cudaEventRecord(start, 0);
		for (float i = 0; i < runs; i++) {
			getVelocity <<<grid1, BLOCK, 0, stream1 >>> (p_GPU, v_GPU);
			getVelocity_Remainder <<<1, remBLOCK, 0, stream2 >>> (p_GPU, v_GPU, offset);
			move <<<fullGrid, BLOCK >>> (p_GPU, v_GPU);
		}
		cudaEventRecord(stop, 0);
		cudaMemcpy(p, p_GPU, N * sizeof(float4), cudaMemcpyDeviceToHost);
		cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
	} else {

		cudaEventRecord(start, 0);
		for (float i = 0; i < runs; i++) {
			getVelocity <<<fullGrid, BLOCK>> > (p_GPU, v_GPU);
			move << <fullGrid, BLOCK >> > (p_GPU, v_GPU);
		}
		cudaEventRecord(stop, 0);
		cudaMemcpy(p, p_GPU, N * sizeof(float4), cudaMemcpyDeviceToHost);
	}


	cudaEventSynchronize(stop); float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n\nGPU time = %0.10f milliseconds\n", elapsedTime / runs);
	cudaError_t cudaEX = cudaGetLastError();
	if (cudaEX != cudaSuccess) {
		printf("cudaError: %s\n", cudaGetErrorString(cudaEX));
	}
	cudaFree(p_GPU); cudaFree(v_GPU);
	cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
	printf("N = %i\n", N);
	set_initail_conditions();
	cuda_NBody(1000);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}