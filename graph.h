#define CPUITERATIONS 96
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
	vdata *devVisited;
	vdata *devResult;
	vdata *result;
};

struct TGraph
{
	int id;
	vdata *graph;
	vdata *devGraph;
	vdata size;
	vdata numarcs;
	vdata *visited;
	int numdevices;
	struct TMemory **memory;
	struct TDeviceSettings **devices;
	cudaStream_t **streams;
};

vdata* file_input(struct TGraph*, char*);
vdata* stdin_input();
int check(vdata*, vdata);
vdata GetVertexCount(struct TGraph*);
vdata GetArcsCount(struct TGraph*);

__global__ void Iteration(vdata*, char*, vdata, vdata*);
int StartIteration(vdata*, char*, vdata, vdata*);