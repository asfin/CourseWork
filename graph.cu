#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include "book.h"

__inline__ __device__ void expand(vdata *graph, vdata *result, vdata a[CPUITERATIONS][SHAREDSIZE], vdata offset , int block, int thread )
{
	vdata vert;
	vdata neigOffset;
	vdata numneig;
	vdata i = 0;

	vert = result[offset];
	neigOffset = graph[vert];
	numneig = graph[neigOffset++];
	a[thread][0] = 0;

	while ((i < numneig)&&(i < SHAREDSIZE-1))
	{
		a[thread][++i] = graph[neigOffset];
	}
	a[thread][0] = i;
	if (i < numneig) 
		offset += block;
	__syncthreads();

}

__inline__ __device__ void compact(vdata a[CPUITERATIONS][SHAREDSIZE], char *visited, int thread)
{
	for (int i = 1; i < a[thread][0]; i++)
	{
		if (visited[a[thread][i]])
			a[thread][i] = a[thread][a[thread][0]--];
	}
}
__global__ void Iteration(vdata *graph, vdata *result, char *visited, vdata size, int ID)
{
	__shared__ vdata a[CPUITERATIONS][SHAREDSIZE];
	vdata tid = blockIdx.x*blockDim.x + threadIdx.x;
	vdata thread = threadIdx.x;
	vdata block = gridDim.x*blockDim.x;

	vdata offset, vert, numneig, neig, i;

	offset = ID*block+tid+1;
	i = 0;

	expand(graph, result, a, offset, block, thread);

	compact(a, visited, thread);
	atomicAdd(&result[0], a[thread][0]);
	__syncthreads();

/*
	while (1)
	{
		vert = devResult[offset];
		numneig = devGraph[devGraph[vert]];
		for (int j = 1; j <= numneig; j++)
		{
			neig = devGraph[devGraph[vert]+j];
			if (!devVisited[neig] && (a[tid][0] < 50))
			{
				//atomicAdd(&devResult[0], 1);
				a[thread][0]++;
				a[thread][a[thread][0]] = neig;
				//devResult[0]++;
				//devResult[devResult[0]] = neig;
				devVisited[neig]++;
			}
			if (a[thread][0] == 50)
			{
				break;
			}
		}
		offset += block;
		
		if (!devResult[offset]) break;
	}*/


	
}


int StartIteration(TGraph *self)
{
	int i = 0, offset = 1, vert, neig, numneig;

	self->result[0] = 1;
	self->result[1] = 0;
	self->visited[0] = 1;

	while (1)
	{
		vert = self->result[offset];
		numneig = self->graph[self->graph[vert]];
		//printf("%d ", vert);
		for (int j = 1; j <= numneig; j++)
		{
			neig = self->graph[self->graph[vert]+j];
			if (!self->visited[neig])
			{
				self->result[2+i++] = neig;
				self->visited[neig]++;
			}
		}
		offset++;
		if (!self->result[offset]) break;
		/*
		if (i > 1000) 
		{
			printf("iteration limit reached.\n");
			break;
		}*/
		if (i >= CPUITERATIONS*BLOCKS*self->numdevices) break;
	}
	self->result[0] += i;

	return 0;
}

vdata* stdin_input()
	{
	vdata len, num, offset, temp, numvertex, numarcs;
	vdata *graph;

	printf("Reading stdin.\n");
		
	scanf("%d %d", &numvertex, &numarcs);
	HANDLE_ERROR(
		cudaHostAlloc((void **)&graph, (2*numvertex+numarcs)*sizeof(vdata), cudaHostAllocWriteCombined|cudaHostAllocMapped)
	);
	offset = 0;
	for (vdata i = 0; i < numvertex; i++)
	{
		scanf("%d %d", &num, &len);
		graph[i] = numvertex + offset;
		graph[numvertex+offset] = len;
		offset += 1;
		for (vdata j = 1; j <= len; j++)
		{
			scanf("%d", &temp);
			graph[numvertex+offset] = temp;
			offset += 1;
		}		
	}
	printf("Graph loaded.\n\n");
	return graph;
}

vdata* file_input(TGraph *self, char *in)
{
	FILE *fp;
	vdata len, num, offset, size;
	vdata *graph, *devGraph;

	fp = fopen(in, "r");		

	fscanf(fp, "%d %d", &self->size, &self->numarcs);
	ERROR(cudaMallocHost((void **)&graph,
						(2*self->size+self->numarcs)*sizeof(vdata),
						cudaHostAllocWriteCombined |
						cudaHostAllocMapped |
						cudaHostAllocPortable
						));
	size = self->size;
	offset = 0;
	for (vdata i = 0; i < size; i++)
	{
		fscanf(fp, "%d %d", &num, &len);
		graph[i] = size + offset;
		graph[size+offset] = len;
		offset += 1;
		for (vdata j = 1; j <= len; j++)
		{
			fscanf(fp, "%d", &graph[size+offset]);
			offset += 1;
		}		
	}

	self->graph = graph;
	return graph;
}

vdata GetVertexCount(struct TGraph *self)
{
	return self->size;
}

vdata GetArcsCount(struct TGraph *self)
{
	return self->numarcs;
}

int check(vdata *graph, vdata size)
{
	vdata t = 0;
	for (vdata ti = 0; ti < size; ti++)
	{
		t += graph[graph[ti]];
	}
	if (t == size)
		return 1;
	else
		return 0;
}