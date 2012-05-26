#include "graph.h"

using namespace std;

extern int pref;
extern int _RUNTHREADS;
int AddKernels(TGraph *self, TDevices *devices)
{
	if (pref)
	{
		FullIteration(self);
		self->byCPU = self->result[0];
		self->total = reduce_host(self->visited, self->size);
		*self->reduced = self->total;
		self->byGPU = 0;
		self->overhead = self->result[0] - self->total;
	}
	else
	{
		StartIteration(self);
		self->byCPU = self->result[0];

		if (self->byCPU >= NUMTHREADS*MAXBLOCKS)
		{
			devices->AddGraphToQuery(self);
		} else {
			self->byGPU = 0;
			self->total = reduce_host(self->visited, self->size);
			*self->reduced = self->total;
			cout << "Insufficent vertex to run kernels.\n";

		}
	}

	return 0;
}

vdata GetReduce(TGraph *self)
{
	cudaSetDevice(self->device);
	cudaDeviceSynchronize();
	self->total = *self->reduced;
	self->byGPU = self->total - self->byCPU;
	self->overhead = self->result[0] - self->total;
	printf("Befor : %d from %d vertex travelled.\n", self->byCPU, self->size);
	printf("After : %d from %d vertex travelled.\n", self->total, self->size);
	printf("By GPU: %d\n", self->byGPU);
	printf("Overhead : %d vertex\n\n", self->overhead);


	return self->byGPU;
}

int TDevices::RunKernels()
{
	int i;
		i = lastused++;
		//i = 0;
		lastused %= numdevices;
			
		cudaSetDevice(i);
		vdata *devred, *buffer;
		for (int j = 0; j < bounded; j++)
		{
			if (boundedGraphs[j]->memOnGPU)
			{
				ERROR(cudaMalloc((void **) &boundedGraphs[j]->devGraph, boundedGraphs[j]->memory.memgraph));
				ERROR(cudaMalloc((void **) &boundedGraphs[j]->devResult, boundedGraphs[j]->memory.memresult));
				ERROR(cudaMalloc((void **) &boundedGraphs[j]->devVisited, boundedGraphs[j]->memory.memvisit));
				ERROR(cudaMemcpyAsync(boundedGraphs[j]->devGraph, boundedGraphs[j]->graph, boundedGraphs[j]->memory.memgraph, cudaMemcpyHostToDevice, devices[i].streams[j]));
				ERROR(cudaMemcpyAsync(boundedGraphs[j]->devResult, boundedGraphs[j]->result, boundedGraphs[j]->memory.memresult, cudaMemcpyHostToDevice, devices[i].streams[j]));
				//ERROR(cudaMemsetAsync(boundedGraphs[j]->devVisited, 0, boundedGraphs[j]->memory.memvisit, devices[i].streams[j]));
				//ERROR(cudaMemcpyAsync(boundedGraphs[j]->devVisited, boundedGraphs[j]->visited, boundedGraphs[j]->memory.memvisit, cudaMemcpyHostToDevice, devices[i].streams[j]));
			} else {
				cout << boundedGraphs[j]->graph << endl;
				ERROR(cudaHostGetDevicePointer(&boundedGraphs[j]->devGraph,   boundedGraphs[j]->graph,   0));
				ERROR(cudaHostGetDevicePointer(&boundedGraphs[j]->devResult,  boundedGraphs[j]->result,  0));
				ERROR(cudaHostGetDevicePointer(&boundedGraphs[j]->devVisited, boundedGraphs[j]->visited, 0));
			}
			ERROR(cudaMalloc((void**)&boundedGraphs[j]->flag, sizeof(char)));
			ERROR(cudaMalloc((void**)&boundedGraphs[j]->lock, sizeof(int)));
			ERROR(cudaMalloc((void**)&boundedGraphs[j]->start, sizeof(vdata)));
			ERROR(cudaHostGetDevicePointer(&boundedGraphs[j]->devReduced,   boundedGraphs[j]->reduced,   0));
			boundedGraphs[j]->device = i;
			
		}
		for (int j = 0; j < bounded; j++)
		{
			InitMem<<<1, 1, 0, devices[i].streams[j]>>>(boundedGraphs[j]->start, boundedGraphs[j]->lock, boundedGraphs[j]->devVisited, boundedGraphs[j]->memory.memvisit);
		}
		for (int j = 0; j < bounded; j++)
		{
				Iteration_cc20<<<MAXBLOCKS, NUMTHREADS, 0, devices[i].streams[j]>>>
						(boundedGraphs[j]->devGraph, boundedGraphs[j]->devResult, boundedGraphs[j]->devVisited, boundedGraphs[j]->start, buffer, boundedGraphs[j]->lock, boundedGraphs[j]->flag);
		}
		for (int j = 0; j < bounded; j++)
		{
			reduce_ccAny<<<1, 512, 0, devices[i].streams[j]>>>(boundedGraphs[j]->devVisited, boundedGraphs[j]->size, boundedGraphs[j]->devReduced);
		}
			//*/
		for (int j = 0; j < bounded; j++)
		{
			if (boundedGraphs[j]->memOnGPU)
			{
				//cudaMemcpy(self->graph, self->devGraph, self->memory.memgraph, cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(boundedGraphs[j]->result, boundedGraphs[j]->devResult, boundedGraphs[j]->memory.memresult, cudaMemcpyDeviceToHost, devices[i].streams[j]);
				cudaMemcpyAsync(boundedGraphs[j]->visited, boundedGraphs[j]->devVisited, boundedGraphs[j]->memory.memvisit, cudaMemcpyDeviceToHost, devices[i].streams[j]);
				cudaFreeAsync(boundedGraphs[j]->devGraph);
				cudaFreeAsync(boundedGraphs[j]->devResult);
				cudaFreeAsync(boundedGraphs[j]->devVisited);
			}
			cudaFreeAsync(boundedGraphs[j]->flag);
			cudaFreeAsync(boundedGraphs[j]->lock);
			cudaFreeAsync(boundedGraphs[j]->start);
		}
	bounded = 0;
	return 0;
}

int TDevices::TransferDataToDevice()
{
	return 0;
}

int TDevices::TransferDataFromDevice()
{
	return 0;
}

int TDevices::AddGraphToQuery(TGraph *self)
{
	boundedGraphs[bounded] = self;
	bounded++;
	if (bounded == maxstreams) RunKernels();
	return 0;
}

TDevices::TDevices()
{
	lastused = 0;
	bounded = 0;
	maxstreams = 4;
	ERROR(cudaGetDeviceCount(&numdevices));
	devices.resize(numdevices);
	boundedGraphs.resize(16);
	for (int i = 0; i < numdevices; i++)
	{
		cudaSetDevice(i);
		devices[i].streams.resize(maxstreams);
		devices[i].DeviceID = i;
		devices[i].laststream = 0;
		cudaGetDeviceProperties(&devices[i].prop, i);
		cudaSetDeviceFlags(cudaDeviceMapHost);
		for (int j = 0; j < maxstreams; j++)
			cudaStreamCreate(&devices[i].streams[j]);
		printf("%s binded.\n", devices[i].prop.name);
	}
	printf("\n");
}

TDevices::~TDevices()
{
	printf("\n");
	for (int i = 0; i < numdevices; i++)
	{
		cudaSetDevice(i);
		ERROR(cudaDeviceReset());
		printf("%s released.\n", devices[i].prop.name);
	}
}

int TDevices::cudaFreeAsync(void *devptr)
{
	devPtrs.push_back(devptr);
	return 0;
}

int TDevices::Clean()
{
	for (int i = 0; i < devPtrs.size(); i++)
	{
		cudaFree(devPtrs.back());
		devPtrs.pop_back();
	}
	return 0;
}
