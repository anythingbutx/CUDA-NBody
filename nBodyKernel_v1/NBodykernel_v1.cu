
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
//gtx 1060: N = 20480
//average time = 16.4179725647 milliseconds

#define BLOCK 1024
#define N 20480 //(1024 * 2) * 10


#define DAMP 0.5f
#define EPSILON 0.000001f
#define DT 0.001f
#define G 1.0f
#define H 1.0f

// Globals
float4 p[N];
float3 v[N], f[N];
float4 *p_GPU;
float3 *v_GPU, *f_GPU;

__global__ void getVelocity(float4 *pos, float3 *vel) {
	const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= N) return;

	const float4 myPos = pos[id];
	float4 force = {0.0f, 0.0f, 0.0f};

	for (int i = 0; i < N; i++) {
		float4 p = pos[i];

		float dx = p.x - myPos.x;
		float dy = p.y - myPos.y;
		float dz = p.z - myPos.z;

		float r2 = dx*dx + dy*dy + dz*dz + EPSILON; 
		float r = 1.0f/sqrtf(r2);
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
}

void cuda_NBody (int runs) {

	cudaSetDevice(0);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	dim3 block, grid;
	block.x = BLOCK;
	block.y = 1; block.z = 1;
	grid.x = (N - 1) / block.x + 1;
	grid.y = 1; grid.z = 1;

	cudaMalloc((void**)&p_GPU, N * sizeof(float4));
	cudaMalloc((void**)&v_GPU, N * sizeof(float3));
	//cudaMalloc((void**)&f_GPU, N * sizeof(float3));

	cudaMemcpy(p_GPU, p, N * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(v_GPU, v, N * sizeof(float3), cudaMemcpyHostToDevice);
	//cudaMemcpy(f_GPU, f, N * sizeof(float3), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	for (float i = 0; i < runs; i++) {
		getVelocity <<<grid, block >>>(p_GPU, v_GPU);
		move <<<grid, block >>>(p_GPU, v_GPU);
		//cudaMemcpy(p, p_GPU, N * sizeof(float4), cudaMemcpyDeviceToHost);
	}

	cudaEventRecord(stop);
	cudaMemcpy(p, p_GPU, N * sizeof(float4), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop); float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	//printf("\n\nblocks:%i X grid:%i\n", block.x, grid.x);
	printf("\n\nGPU time = %0.10f milliseconds\n", elapsedTime/runs);
	cudaFree(p_GPU); cudaFree(v_GPU);
	cudaEventDestroy(start); cudaEventDestroy(stop);
	//cudaDeviceReset();
}

int main() {
	set_initail_conditions();
	cuda_NBody(1000);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}
