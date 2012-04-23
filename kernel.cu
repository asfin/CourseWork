#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include "graph.h"
#include "book.h"
#include <iostream>

using namespace std;

float	Run_Kernels(TGraph*);
int		InitMemory(TGraph*);
int		InitDeviceSettings(TGraph*);
void	PrintStat(TGraph*);
void	ReleaseCards(TGraph*);
int		Create_Graph(TGraph*, char path[256], int id=0);

int main(int argc, char* argv[])
{
	cout << "Graph traversal on GPU.\n\n";

	vdata size, memgraph;
	float time;
	vector<TGraph> graph;
	int total = 6, num = 0;
	graph.resize(total-num);
	char open[256];

	while (num < total)
	{
		//sprintf((char*)open, "C:\\graphs\\input%d.txt", num);
		sprintf((char*)open, "\\\\FILE-SERVER\\raid_root\\graphs\\input%d.txt", num+3);
		Create_Graph(&graph[num], open);
		PrintStat(&graph[num]);
		StartIteration(&graph[num]);
		num++;
	}
	num = 0;
	while (num < total)
	{
		
		printf("%d from %d vertex travelled.\n", graph[num].result[0], graph[num].size);

		if (graph[num].result[0] >= CPUITERATIONS*BLOCKS*graph[num].numdevices)
		{
			//time = Run_Kernels(&graph[num]);
			//printf("Iterations completed in %.3fms\n", time);
			for (int i = 0; i < graph[num].numdevices; i++)
			{
				cudaSetDevice(i);
				ERROR(
				cudaMemcpyAsync(graph[num].devices[i].devVisited, graph[num].visited, graph[num].memory[i+1].memvisit, cudaMemcpyHostToDevice, graph[0].devices[i].stream)
				);
				Iteration<<<BLOCKS, CPUITERATIONS, 0, graph[0].devices[i].stream>>>
					(graph[num].devices[i].devGraph, graph[num].devices[i].devResult, graph[num].devices[i].devVisited, graph[num].size, graph[num].devices[i].DeviceID);
			}
		} else {
			cout << "Insufficent vertex to run kernels.\n";
		}

		printf("%d from %d vertex travelled.\n", graph[num].result[0], graph[num].size);
	

		
		num++;
	}
	ReleaseCards(&graph[0]);
    return 0;
}

float Run_Kernels(TGraph *self)
{
	float timer;
	cudaEvent_t start, stop;

	HANDLE_ERROR(
		cudaEventCreate(&start)
		);
	HANDLE_ERROR(
		cudaEventCreate(&stop)
		);
	
	for (int i = 0; i < self->numdevices; i++)
	{
		cudaSetDevice(i);
		ERROR(
		cudaMemcpyAsync(self->devices[i].devVisited, self->visited, self->memory[i+1].memvisit, cudaMemcpyHostToDevice, self->devices[i].stream)
		);
	}
	HANDLE_ERROR(
		cudaEventRecord(start, 0)
		);
	
	for (int i = 0; i < self->numdevices; i++)
	{
		cudaSetDevice(i);
		//cudaMemcpyAsync(self->devices[i].devResult+1, self->result+1+i* );
		Iteration<<<BLOCKS, CPUITERATIONS, 0, self->devices[i].stream>>>
			(self->devices[i].devGraph, self->devices[i].devResult, self->devices[i].devVisited, self->size, self->devices[i].DeviceID);
	}
	cudaDeviceSynchronize();
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

void PrintStat(TGraph *self)
{
	printf("\nsize of vdata : %d\n",		sizeof(vdata));
	printf("vertex in graph : %d\n",		self->size);
	printf("arcs in graph   : %d\n\n",		GetArcsCount(self));
	printf("size of graph   : %3.3fMb\n",	(float)self->memory[0].memgraph/1048576);
	printf("size of visited : %3.3fMb\n",	(float)self->memory[0].memvisit/1048576);
	printf("size of result  : %3.3fMb\n",	(float)self->memory[0].memresult/1048576);
	printf("Total allocated : %3.3fMb\n\n",	(float)(self->memory[0].memgraph+self->memory[0].memvisit+self->memory[0].memresult)/1048576);
}

int InitMemory(TGraph *self)
{
	self->memory.resize(self->numdevices+1);
	vdata checkvis = 0, checkres = 0;

	self->memory[0].memgraph  = (GetArcsCount(self)+2*GetVertexCount(self))*sizeof(vdata);
	self->memory[0].memvisit  =  GetVertexCount(self)*sizeof(char);
	self->memory[0].memresult = (GetVertexCount(self)+1+self->numdevices)*sizeof(vdata);
	
	for (int i = 1; i < self->numdevices+1; i++)
	{
		self->memory[i].memgraph  = self->memory[0].memgraph;
		self->memory[i].memvisit  = (GetVertexCount(self))*sizeof(char);
		//self->memory[i].memresult = GetVertexCount(self)*sizeof(vdata)/self->numdevices+sizeof(vdata);

		checkvis += self->memory[i].memvisit;
		//checkres += self->memory[i].memresult;
	}
	//self->memory[1].memresult += self->memory[0].memresult-checkres;
	//self->memory[1].memvisit  += self->memory[0].memvisit-checkvis;
	self->memory[0].memvisit = checkvis;

	return 0;

}

int InitDeviceSettings(TGraph *self)
{
	cudaDeviceProp prop;

	self->devices.resize(self->numdevices);

	cudaSetDevice(0);
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaGetDeviceProperties(&prop, 0);
	cudaStreamCreate(&(self->devices[0].stream));
	self->devices[0].DeviceID = 0;
	self->devices[0].start    = 0;
	self->devices[0].stop     = self->size/self->numdevices+self->size%self->numdevices;
	/*ERROR(cudaHostAlloc((void **) &(self->devices[0].result),
						self->memory[1].memresult,
						cudaHostAllocWriteCombined|cudaHostAllocMapped)
						);
	ERROR(cudaHostGetDevicePointer(&(self->devices[0].devResult), self->devices[0].result, 0));*/
	ERROR(cudaHostGetDevicePointer(&(self->devices[0].devResult), self->result, 0));
	ERROR(cudaHostGetDevicePointer(&self->devices[0].devGraph, self->graph, 0));
	
	self->devices[0].name = (char*)malloc(256);
	sprintf(self->devices[0].name, "%s", prop.name);
	
	ERROR(cudaMalloc((void **) &(self->devices[0].devVisited), self->memory[1].memvisit));
	printf("%s binded.\n", self->devices[0].name);

	for (int i = 1; i < self->numdevices; i++)
	{
		cudaSetDevice(i);
		cudaGetDeviceProperties(&prop, i);
		cudaSetDeviceFlags(cudaDeviceMapHost);
		cudaStreamCreate(&(self->devices[i].stream));
		self->devices[i].DeviceID = i;
		self->devices[i].start    = self->devices[i-1].stop+1;
		self->devices[i].stop     = self->devices[i].start+self->size/self->numdevices;
		/*ERROR(cudaHostAlloc((void **) &(self->devices[i].result),
						self->memory[i+1].memresult,
						cudaHostAllocWriteCombined|cudaHostAllocMapped)
						);
		ERROR(cudaHostGetDevicePointer(&(self->devices[i].devResult), self->devices[i].result, 0));*/
		ERROR(cudaHostGetDevicePointer(&(self->devices[i].devResult), self->result, 0));
		self->devices[i].name = (char*)malloc(256);
		sprintf(self->devices[i].name, "%s", prop.name);
		ERROR(cudaMalloc((void **) &(self->devices[i].devVisited), self->memory[i+1].memvisit));
		ERROR(cudaHostGetDevicePointer(&self->devices[i].devGraph, self->graph, 0));
		printf("%s binded.\n", self->devices[i].name);
	}

	return 0;
}

int Create_Graph(TGraph *self, char path[256], int id)
{
	self->id = id;
	cudaSetDevice(0);
	printf("Opening %s\n", path);
	file_input(self, path);
	printf("Graph loaded.\n\n");
	
	ERROR(cudaGetDeviceCount(&self->numdevices));

	InitMemory(self);

	ERROR(cudaHostAlloc((void **) &self->result,
						//CPUITERATIONS*BLOCKS*self->numdevices+1,
						self->memory[0].memresult,
						cudaHostAllocWriteCombined|
						cudaHostAllocMapped|
						cudaHostAllocPortable
						));

	ERROR(cudaHostAlloc((void **) &(self->visited), self->memory[0].memvisit, cudaHostAllocWriteCombined|cudaHostAllocMapped));

	InitDeviceSettings(self);

	return 0;
}

void ReleaseCards(TGraph *self)
{
	cudaDeviceProp prop;
	for (int i = 0; i < self->numdevices; i++)
	{
		cudaSetDevice(i);
		cudaGetDeviceProperties(&prop, i);
		ERROR(cudaStreamDestroy(self->devices[i].stream));
		ERROR(cudaDeviceReset());
		printf("%s released.\n", prop.name);
	}
}