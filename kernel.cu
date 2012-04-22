#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "graph.h"
#include "book.h"
#include <iostream>

using namespace std;

float Run_Kernel(vdata*, char*, vdata, vdata*);
struct TMemory** InitMemory(struct TMemory **);
struct TDeviceSettings** InitDeviceSettings(struct TGraph*);
void PrintStat(struct TGraph *self);
void ReleaseCards(struct TGraph *self);
struct TGraph Create_Graph(char path[256], int id=0);

__global__ void test(int *b, int i)
{
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	b[i+tid] = 1;
	for (int t = 0; t < 65535; t++){}
}

int main(int argc, char* argv[])
{
	cout << "Graph traversal on GPU.\n\n";

	vdata size, memgraph;
	struct TGraph graph;
	char open[256];

	sprintf((char*)open, "C:\\graphs\\input%d.txt", 0);
	//sprintf((char*)open, "\\\\FILE-SERVER\\raid_root\\graphs\\input%d.txt", 0);
	graph = Create_Graph(open);
	PrintStat(&graph);
		
	int *a, *deva;
	cudaSetDevice(0);
	ERROR(cudaMallocHost((void **)&a,
						20*sizeof(int),
						cudaHostAllocWriteCombined |
						cudaHostAllocMapped |
						cudaHostAllocPortable
						));
	cudaHostGetDevicePointer(&deva, a, 0);
	for (int i = 0; i < graph.numdevices; i++)
	{
		cudaSetDevice(i);
		test<<<2, 2, 0, *graph.streams[i] >>>(deva, i*5);
	}
	cudaDeviceSynchronize();
	for (int i = 0; i < 10; i++)
		printf("%d ", a[i]);

	//ERROR(cudaMalloc((void **) &devGraph, (2*GetVertexCount()+GetArcsCount())*sizeof(vdata)));
	
	//ERROR(cudaMemcpy(devGraph, graph, (2*GetVertexCount()+GetArcsCount())*sizeof(vdata), cudaMemcpyHostToDevice));


	
	//ERROR(cudaHostGetDevicePointer(&devVisited, visited, 0));

	//ERROR(cudaHostAlloc((void **) &result, memresult, cudaHostAllocWriteCombined|cudaHostAllocMapped));
	//ERROR(cudaHostGetDevicePointer(&devResult, result, 0));
	
	
	/*
	StartIteration(graph, visited, GetVertexCount(), result);
	printf("%d from %d vertex travelled.\n", result[0], GetVertexCount());
	ERROR(cudaMemcpy(devVisited, visited, memvisit, cudaMemcpyHostToDevice));
	
	printf("Iterations completed in %.3fms\n", Run_Kernel(devGraph, devVisited, GetVertexCount(), devResult));

	printf("%d from %d vertex travelled.\n", result[0], GetVertexCount());
	*/

	ReleaseCards(&graph);
	//scanf("%s", &open);
    return 0;
}

float Run_Kernel(vdata *devGraph, char *devVisited, vdata size, vdata* devResult)
{
	float timer;
	cudaEvent_t start, stop;

	HANDLE_ERROR(
		cudaEventCreate(&start)
		);
	HANDLE_ERROR(
		cudaEventCreate(&stop)
		);
	HANDLE_ERROR(
		cudaEventRecord(start, 0)
		);

	Iteration<<<1, CPUITERATIONS>>>(devGraph, devVisited, size, devResult);

	HANDLE_ERROR(
		cudaEventRecord(stop, 0)
		);
	HANDLE_ERROR(
		cudaEventSynchronize(stop)
		);
	HANDLE_ERROR(
		cudaEventElapsedTime(&timer, start, stop)
		);

	return timer;
}

void PrintStat(struct TGraph *self)
{
	printf("\nsize of vdata : %d\n", sizeof(vdata));
	printf("vertex in graph : %d\n",  self->size);
	printf("arcs in graph   : %d\n\n", GetArcsCount(self));
	printf("size of graph   : %3.3fMb\n", (float)self->memory[0]->memgraph/1048576);
	printf("size of visited : %3.3fMb\n", (float)self->memory[0]->memvisit/1048576);
	printf("size of result  : %3.3fMb\n", (float)self->memory[0]->memresult/1048576);
	printf("Total allocated : %3.3fMb\n\n", (float)(self->memory[0]->memgraph+self->memory[0]->memvisit+self->memory[0]->memresult)/1048576);
}

struct TMemory** InitMemory(struct TGraph *self)
{
	struct TMemory **mem;
	vdata checkvis = 0, checkres = 0;

	mem = (struct TMemory **) malloc((self->numdevices+1)*sizeof(struct TMemory*));
	mem[0] = (struct TMemory *) malloc(sizeof(struct TMemory));
	mem[0]->memgraph  = (GetArcsCount(self)+2*GetVertexCount(self))*sizeof(vdata);
	mem[0]->memvisit  =  GetVertexCount(self)*sizeof(char);
	mem[0]->memresult = (GetVertexCount(self)+1+self->numdevices)*sizeof(vdata);
	
	for (int i = 1; i < self->numdevices+1; i++)
	{
		mem[i] = (struct TMemory *) malloc(sizeof(struct TMemory));
		mem[i]->memgraph  = mem[0]->memgraph;
		mem[i]->memvisit  = (GetVertexCount(self)/self->numdevices)*sizeof(char);
		mem[i]->memresult = GetVertexCount(self)*sizeof(vdata)/self->numdevices+sizeof(vdata);

		checkvis += mem[i]->memvisit;
		checkres += mem[i]->memresult;
	}
	mem[1]->memresult += mem[0]->memresult-checkres;
	mem[1]->memvisit  += mem[0]->memvisit-checkvis;

	return mem;

}

struct TDeviceSettings** InitDeviceSettings(struct TGraph *self)
{
	cudaDeviceProp prop;
	static struct TDeviceSettings **pdev;

	pdev = (struct TDeviceSettings **) malloc(self->numdevices*sizeof(struct TDeviceSettings*));
	pdev[0] = (struct TDeviceSettings *) malloc(sizeof(struct TDeviceSettings));
	self->streams = (cudaStream_t **) malloc(sizeof(cudaStream_t *));
	self->streams[0] = (cudaStream_t *) malloc(sizeof(cudaStream_t));

	cudaSetDevice(0);
	cudaGetDeviceProperties(&prop, 0);
	cudaStreamCreate(self->streams[0]);
	pdev[0]->DeviceID = 0;
	pdev[0]->start    = 0;
	pdev[0]->stop     = self->size/self->numdevices+self->size%self->numdevices;
	ERROR(cudaHostAlloc((void **) &(pdev[0]->result),
						self->memory[1]->memresult,
						cudaHostAllocWriteCombined|cudaHostAllocMapped)
						);
	ERROR(cudaHostGetDevicePointer(&(pdev[0]->devResult), pdev[0]->result, 0));
	pdev[0]->devGraph = self->devGraph;
	
	pdev[0]->name = (char*)malloc(256);
	sprintf(pdev[0]->name, "%s", prop.name);
	cudaSetDeviceFlags(cudaDeviceMapHost);
	ERROR(cudaMalloc((void **) &(pdev[0]->devVisited), self->memory[1]->memvisit));
	printf("%s binded.\n", pdev[0]->name);

	for (int i = 1; i < self->numdevices; i++)
	{
		cudaSetDevice(i);
		cudaGetDeviceProperties(&prop, i);
		cudaSetDeviceFlags(cudaDeviceMapHost);
		pdev[i] = (struct TDeviceSettings *)malloc(sizeof(struct TDeviceSettings));
		self->streams[i] = (cudaStream_t *) malloc(sizeof(cudaStream_t));
		cudaStreamCreate(self->streams[i]);
		pdev[i]->DeviceID = i;
		pdev[i]->start    = pdev[i-1]->stop+1;
		pdev[i]->stop     = pdev[i]->start+self->size/self->numdevices;
		ERROR(cudaHostAlloc((void **) &(pdev[i]->result),
						self->memory[i+1]->memresult,
						cudaHostAllocWriteCombined|cudaHostAllocMapped)
						);
		ERROR(cudaHostGetDevicePointer(&(pdev[i]->devResult), pdev[i]->result, 0));
		pdev[i]->name = (char*)malloc(256);
		sprintf(pdev[i]->name, "%s", prop.name);
		ERROR(cudaMalloc((void **) &(pdev[i]->devVisited), self->memory[i+1]->memvisit));
		pdev[i]->devGraph = pdev[0]->devGraph;
		printf("%s binded.\n", pdev[i]->name);
	}

	return pdev;
}

struct TGraph Create_Graph(char path[256], int id)
{
	struct TGraph self;
	self.id = id;
	//cudaSetDevice(0);
	printf("Opening %s\n", path);
	self.graph = file_input(&self, path);
	printf("Graph loaded.\n\n");
	
	ERROR(cudaGetDeviceCount(&self.numdevices));

	self.memory  = InitMemory(&self);

	ERROR(cudaHostAlloc((void **) &self.visited, self.memory[0]->memvisit, cudaHostAllocWriteCombined|cudaHostAllocMapped));

	self.devices = InitDeviceSettings(&self);

	return self;
}

void ReleaseCards(struct TGraph *self)
{
	for (int i = 0; i < self->numdevices; i++)
	{
		cudaSetDevice(i);
		ERROR(cudaFree(self->devices[i]->devVisited));
		ERROR(cudaStreamDestroy(*(self->streams[i])));
		ERROR(cudaDeviceReset());
		printf("%s released.\n", self->devices[i]->name);
	}
}