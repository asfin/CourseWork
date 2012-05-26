#include "graph.h"

using namespace std;

__device__ void prefsum(unsigned char *in, unsigned int *out, int max)
{
	out[0] = 0;
	for (int i = 1; i < max+1; i++)
		out[i] = out [i-1] + in[i-1];
}

__global__ void Iteration_cc20(vdata *graph, vdata *result, char *visited, vdata *start, vdata *buffer, int *lock, char *flag)

{
	__shared__ vdata a[NUMTHREADS][SHAREDSIZE];
	__shared__ unsigned char len[NUMTHREADS];
	__shared__ unsigned int  sum[NUMTHREADS+1];
	__shared__ vdata to, from, notload;

	vdata thread = threadIdx.x;

	int i, t;
	
	vdata numneig;
	vdata vert, neigOffset;

	if (thread == 0) 
	{
		notload = 0;
	}
	int walker = 0;
	
st:
	__syncthreads();

	if (notload == 0)
	{
		if (thread == 0)
		{
			while (atomicCAS(lock, 0, 1) != 0);
			from = (*start);
			if (result[0] > *start + NUMTHREADS)
			{
				*start += NUMTHREADS;
			} else {
				*start = result[0];
			}
			to = (*start);
			atomicExch(lock, 0);
		}
		__syncthreads();

		vert = 0;
		if (from+thread <= to)
			vert = result[from+thread];
		neigOffset = graph[vert];
		numneig = graph[neigOffset++];
	} else {
		notload = 0;
		if (walker == numneig)
			numneig = 0;
		else if (numneig > SHAREDSIZE)
			numneig -= SHAREDSIZE;
	}
	//__syncthreads();
	if (numneig > SHAREDSIZE)
		notload = 1;
	walker = 0;

	

	while (walker < numneig && walker < SHAREDSIZE)
	{
		a[thread][walker] = graph[neigOffset];
		neigOffset++;
		walker++;
	}

	len[thread] = walker;

	for (i = 0; i < len[thread]; i++)
	{
		if (visited[a[thread][i]])
		{
			a[thread][i--] = a[thread][--len[thread]];
		} else {
			visited[a[thread][i]] = 1;
		}
	}

	if (thread == 0)
	{
		prefsum(len, sum, NUMTHREADS);
	}
	__syncthreads();

	if (sum[NUMTHREADS])
	{
		if (thread == 0) while (atomicCAS(lock, 0, 1) != 0);
		__syncthreads();
		t = result[0]+sum[thread]+1;
		
		for (i = 0; i < len[thread]; i++)
		{
			result[t+i] = a[thread][i];
		}
		__syncthreads();
		if (thread == 0) 
		{
			atomicAdd(&result[0], sum[NUMTHREADS]);	
			atomicExch(lock, 0);
		}
		__syncthreads();
	}
	if (*start < result[0] || sum[NUMTHREADS]) goto st;
	
	//if (iteration++ < 3) goto st;
	thread++;

}
/*
__global__ void Iteration_bred(vdata *graph, vdata *result, char *visited, vdata size, vdata *buffer, int *lock, char *flag)
{
	int i = 0, offset = 1, vert, neig, numneig;

	result[0] = 1;
	result[1] = 0;
	visited[0] = 1;

	while (1)
	{
		vert = result[offset];
		numneig = graph[graph[vert]];
		for (int j = 1; j <= numneig; j++)
		{
			neig = graph[graph[vert]+j];
			if (!visited[neig])
			{
				result[2+i++] = neig;
				visited[neig]++;
			}
		}
		offset++;
		if (!result[offset]) break;
		//if (i >= NUMTHREADS*MAXBLOCKS) break;
	}
	result[0] += i;
}
*/
__global__ void reduce_ccAny(char *in, vdata size, vdata *out)
{
	__shared__ vdata temp[512];
	vdata tid = blockDim.x*blockIdx.x+threadIdx.x;
	vdata thread = threadIdx.x;
	vdata block = blockDim.x*gridDim.x;
	vdata pos = tid;

	temp[thread] = 0;
	//if (tid == 0) 
	//	*out = 0;

	while (pos < size)
	{
		if (in[pos])
			temp[thread]++;
		pos += block;
	}
	__syncthreads();

	int i = blockDim.x/2;
	while (i > 0)
	{
		if (thread < i)
			temp[thread] += temp[thread+i];
		__syncthreads();
		i /= 2;
	}

	if (thread == 0)
	{
		atomicExch(out, temp[0]);
	}

}

__global__ void InitMem(vdata *start, int *lock, char *vis, vdata size)
{
	*start = 1;
	*lock = 0;
	for (vdata i = 0; i < size; i++)
		vis[i] = 0;
}
