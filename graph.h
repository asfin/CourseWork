#include <cuda_runtime.h>
#include <sm_11_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>

#include "book.h"

#define NUMTHREADS	224
#define MAXBLOCKS	1
#define SHAREDSIZE	15
// 192 - threads
// 15 - shared
// optimal for 1 block
#define MAXGRAPHS	16
typedef unsigned int vdata;

struct TMemory
{
	vdata memgraph;
	vdata memvisit;
	vdata memresult;
};
struct TDevice
{
	int DeviceID;
	cudaDeviceProp prop;
	int laststream;
	std::vector<cudaStream_t> streams;
	
};

struct TGraph
{
	int		id;
	char	meta[256];
	vdata	*graph, *devGraph;
	vdata	*result, *devResult;
	char	*visited, *devVisited;
	vdata	size;
	vdata	numarcs;
	vdata	*reduced;
	vdata	*devReduced;
	char	*flag;
	int		*lock;
	vdata	*start;
	int		memOnGPU;
	int		device;
	vdata	byCPU;
	vdata	byGPU;
	vdata	total;
	vdata	overhead;
	TMemory	memory;
};

class TDevices
{
private:
	int	numdevices;
	int lastused;
	int bounded;
	int maxstreams;
	std::vector<TGraph*>	boundedGraphs;
	std::vector<TDevice>	devices;
	std::vector<void*>		devPtrs;
public:
	TDevices();
	int AddGraphToQuery(TGraph*);
	int RunKernels();
	int cudaFreeAsync(void *devptr);
	int Clean();
	int TransferDataToDevice();
	int TransferDataFromDevice();
	~TDevices();
};



vdata*	file_input(struct TGraph*, char*);
vdata*	stdin_input();

int		check(vdata*, vdata);
vdata	GetVertexCount(struct TGraph*);
vdata	GetArcsCount(struct TGraph*);

__global__
void	Iteration_cc11(vdata*, vdata*, char*, vdata*, vdata*, int*, char*);
__global__
void	Iteration_cc20(vdata*, vdata*, char*, vdata*, vdata*, int*, char*);
int		StartIteration(TGraph*);
int		FullIteration(TGraph*);
int		AddKernels(TGraph*, TDevices*);
vdata	GetReduce(TGraph*);

int		InitMemory(TGraph*);
void	PrintMemStat(TGraph*);
int		PrintStat(TGraph*, char*);
int		Create_Graph(TGraph*, char path[256], int id=0);
int		Destroy_Graph(TGraph*);
//vdata reduce(char*, vdata, cudaStream_t);
vdata reduce_host(char *in, vdata size);
__global__ void reduce_ccAny(char *in, vdata size, vdata *out);
__global__ void InitMem(vdata *start, int *lock, char*, vdata);
__global__
void	Iteration_bred(vdata*, vdata*, char*, vdata, vdata*, int*, char*);
