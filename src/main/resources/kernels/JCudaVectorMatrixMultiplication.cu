extern "C"
__global__ void multiplication(int* M, int* N, int* P, int Width)
{
    int tid, tx, ty;
	tx = blockDim.x*blockIdx.x + threadIdx.x;
	ty = blockDim.y*blockIdx.y + threadIdx.y;
	tid = Width*ty + tx;
       
   	int Value = 0;
	int MVal = 0;
	int NVal = 0;

	for (int i = 0; i < Width; i++)
	{
		MVal = M[ty * Width + i];
		NVal = N[i * Width + tx];
		Value += MVal * NVal;
	}

	P[tid] = Value;

}