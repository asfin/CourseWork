#include <vector>

#define CPUITERATIONS 96
#define BLOCKS 7
typedef unsigned long vdata;

struct TMemory
{
	vdata memgraph;
	vdata memvisit;
	vdata memresult;
};
struct TDeviceSettings
{
	int DeviceID;
	char *name;
	vdata start;
	vdata stop;
	vdata *devGraph;
	char *devVisited;
	vdata *devResult;
	vdata *result;
	cudaStream_t stream;
};

struct TGraph
{
	int id;
	vdata *graph;
	vdata size;
	vdata numarcs;
	vdata *result;
	char *visited;
	int numdevices;
	std::vector<TMemory> memory;
	std::vector<TDeviceSettings> devices;
};

vdata* file_input(struct TGraph*, char*);
vdata* stdin_input();
int check(vdata*, vdata);
vdata GetVertexCount(struct TGraph*);
vdata GetArcsCount(struct TGraph*);

__global__ void Iteration(vdata*, char*, vdata, vdata*);
int StartIteration(TGraph*);