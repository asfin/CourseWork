#include "graph.h"

using namespace std;

int pref;

int main(int argc, char* argv[])
{
	char open[256], stat[256], folder[256], folderIn[256], folderOut[256];

	
	int	total	=	32	;
	int	num		=	0	;
	int	from	=	0	;
	int	step	=	16	;
		pref	=	0	;

	vector<TGraph>	graph;
	TDevices		devices;
	fstream			file;

	if (argc >= 2)
	{
		sprintf((char*)folder, argv[1]);
		cout << folder << endl;
		sprintf((char*)folderIn, folder, "input%d.txt");
		sprintf((char*)folderOut, folder, "stat%d.txt");
	}
	else
	{
		sprintf((char*)folderIn, "/media/2TB/graphs/0.001440/input%%d.txt");
		sprintf((char*)folderOut, "/media/2TB/graphs/0.001440/stat%%d.txt");
		//sprintf((char*)folderIn, "D:\\graphs\\0.001430\\input%%d.txt");
		//sprintf((char*)folderOut, "D:\\graphs\\0.001430\\stat%%d.txt");
	}
	if (argc >= 3)
		total = atoi(argv[2]);
	if (argc >= 4)
		from  = atoi(argv[3]);
	if (argc >= 5)
		pref  = atoi(argv[4]);

	if (pref) 
		cout << "Graph traversal on CPU\n";
	else
		cout << "Graph traversal on GPU\n";

	//graph.resize(total);
	graph.resize(128);
	int t = from+total;
	for (from; from < t; from += step)
	{
		for (num = from; num < from + step; num++)
		{
			sprintf((char*)open, folderIn, num);
			//sprintf((char*)open, "\\\\FILE-SERVER\\raid_root\\graphs\\input%d.txt", num);
			Create_Graph(&graph[num], open, num);
			//PrintMemStat(&graph[num]);
		}
		
		for (num = from; num < from + step; num++)
		{
			AddKernels(&graph[num], &devices);
		}
		if (!pref)
		{
			devices.RunKernels();
			devices.Clean();
		}
		for (num = from; num < from + step; num++)
		{
			if (!pref)
			GetReduce(&graph[num]);
			Destroy_Graph(&graph[num]);
		}
	
		for (num = from; num < from + step; num++)
		{
			//sprintf((char*)stat, "C:\\pypy-1.8\\graphs\\stat%d.txt", num);
			sprintf((char*)open, folderOut, num);
			PrintStat(&graph[num], open);
			//sprintf((char*)open, "\\\\FILE-SERVER\\raid_root\\graphs\\stat%d.txt", num);
			
		}
	}
		/*
		file.open("C:\\graphs\\debug.txt", ios::out);
		for (int i = 0; i <= graph[0].result[0]; i++)
		{
			file << graph[0].result[i]<<' ';
			if (!(i%20)) file << '\n';
		}
		file.close();
	
	
	//*/
    return 0;
}

int	PrintStat(TGraph *self, char *open)
{
	fstream file;
	file.open(open, ios::out|ios::app);
	file << "total=" << self->total << " ";
	file << "byCPU=" << self->byCPU << " ";
	file << "byGPU=" << self->byGPU << " ";
	file << "overhead=" << self->overhead << " " << endl;// << " 2 blocks - production ready";
	file.close();
	return 0;
}
