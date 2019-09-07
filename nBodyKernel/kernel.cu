
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Optimized using shared memory and on chip memory																																												
// nvcc nbodyGPU5.cu -o GPU5 -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//gtx 1060: N = 20480, BLOCK = 1024
//average time = 95.8 milliseconds


#define N 20480
#define BLOCK 1024

#define DAMP 0.5

#define DT 0.001
#define STOP_TIME 1.0

#define G 1.0
#define H 1.0

// Globals
float4 p[N];
float3 v[N], f[N];
float4 *p_GPU;
float3 *v_GPU, *f_GPU;
dim3 block, grid;

void set_initail_conditions()
{
	int i, j, k, num, particles_per_side;
	float position_start, temp;
	float initail_seperation;

	temp = pow((float)N, 1.0 / 3.0) + 0.99999;
	particles_per_side = temp;
	printf("\n cube root of N = %d \n", particles_per_side);
	position_start = -(particles_per_side - 1.0) / 2.0;
	initail_seperation = 2.0;

	for (i = 0; i<N; i++)
	{
		p[i].w = 1.0;
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

	block.x = BLOCK;
	block.y = 1;
	block.z = 1;

	grid.x = (N - 1) / block.x + 1;
	grid.y = 1;
	grid.z = 1;

	cudaMalloc((void**)&p_GPU, N * sizeof(float4));
	cudaMalloc((void**)&v_GPU, N * sizeof(float3));
	cudaMalloc((void**)&f_GPU, N * sizeof(float3));
}

__device__ float3 getBodyBodyForce(float4 p0, float4 p1)
{
	float3 f;
	float dx = p1.x - p0.x;
	float dy = p1.y - p0.y;
	float dz = p1.z - p0.z;
	float r2 = dx*dx + dy*dy + dz*dz;
	float r = sqrt(r2);

	float force = (G*p0.w*p1.w) / (r2)-(H*p0.w*p1.w) / (r2*r2);

	f.x = force*dx / r;
	f.y = force*dy / r;
	f.z = force*dz / r;

	return(f);
}

__global__ void getForces(float4 *pos, float3 *vel, float3 * force)
{
	int j, ii;
	float3 force_mag, forceSum;
	float4 posMe;
	__shared__ float4 shPos[BLOCK];
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	forceSum.x = 0.0;
	forceSum.y = 0.0;
	forceSum.z = 0.0;

	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
	posMe.z = pos[id].z;
	posMe.w = pos[id].w;

	for (j = 0; j < gridDim.x; j++)
	{
		shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
		__syncthreads();

		#pragma unroll 32
		for (int i = 0; i < blockDim.x; i++)
		{
			ii = i + blockDim.x*j;
			if (ii != id && ii < N)
			{
				force_mag = getBodyBodyForce(posMe, shPos[i]);
				forceSum.x += force_mag.x;
				forceSum.y += force_mag.y;
				forceSum.z += force_mag.z;
			}
		}
	}
	if (id <N)
	{
		force[id].x = forceSum.x;
		force[id].y = forceSum.y;
		force[id].z = forceSum.z;
	}
}

__global__ void moveBodies(float4 *pos, float3 *vel, float3 * force)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if (id < N)
	{
		vel[id].x += ((force[id].x - DAMP*vel[id].x) / pos[id].w)*DT;
		vel[id].y += ((force[id].y - DAMP*vel[id].y) / pos[id].w)*DT;
		vel[id].z += ((force[id].z - DAMP*vel[id].z) / pos[id].w)*DT;

		pos[id].x += vel[id].x*DT;
		pos[id].y += vel[id].y*DT;
		pos[id].z += vel[id].z*DT;
	}
}

void n_body(int runs)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(p_GPU, p, N * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(v_GPU, v, N * sizeof(float3), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start);
	for(int i = 0; i < runs; i++) {
		getForces <<<grid, block >>>(p_GPU, v_GPU, f_GPU);
		moveBodies <<<grid, block >>>(p_GPU, v_GPU, f_GPU);
		
	}
	cudaEventRecord(stop);
	cudaMemcpy(p, p_GPU, N * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n\nGPU time = %3.1f milliseconds\n", elapsedTime/runs);
	cudaFree(p_GPU); cudaFree(v_GPU); cudaFree(f_GPU);
	cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char** argv)
{
	cudaSetDevice(0);

	set_initail_conditions();
	n_body(1000);

	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}