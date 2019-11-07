#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cufft.h>
#include <Windows.h>
#include <vector>
#include <ctime>
#include <omp.h>

#define SAMPLERATE 8
// cuda defs
#define BLOCK_COUNT 16
#define THREAD_COUNT 128


// using stds
using std::ios_base;
using std::ifstream;
using std::ofstream;
using std::string;
using std::cout;
using std::endl;
using std::pow;
using std::ios;
using std::vector;
using std::distance;
using std::clock;
//
cufftComplex* audioFFTCuda(int* array, int start, int end);
float cosineSimilarity(float* a, float* b, int sampleSize, int offset, int audioSize);
float cosineSimilarity(float* a, float* b, int sampleSize);
float cosineSimilarityCuda(float* a, float* b, int sampleSize);
float ladCuda(float* a, float* b, int sampleSize);
float* normalize(cufftComplex* c, int size);
float findMax(float* a, int N);
float findMaxC(cufftComplex* a, int N);
float* normalizeCuda(cufftComplex* array, int size);
float findMin(float* a, int N);
float findMinC(cufftComplex* a, int N);
float lad(float* a, float* b, int sampleSize);
float findMaxCuda(float* array, int size);
float findMinCuda(float* array, int size);
int* readAudio(const char* path, int& size);
int countWords(std::ifstream& in);
void compare(string dirPath, string samplePath);
vector<string> get_all_files_names_within_folder(string folder);

__global__ void cosineKernel(float *a, float *b, float *outN, float *outD1, float *outD2, int size) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	while (i < size) {
		sdata[3 * tid] = a[i] * b[i] + a[i + blockDim.x] * b[i + blockDim.x];
		sdata[3 * tid + 1] = a[i] * b[i] + a[i + blockDim.x] * b[i + blockDim.x];
		sdata[3 * tid + 2] = a[i] * b[i] + a[i + blockDim.x] * b[i + blockDim.x];
		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 96; s >>= 1) {
			if (tid < s) {
				sdata[3 * tid] += sdata[3 * tid + s];
				sdata[3 * tid + 1] += sdata[3 * tid + s + 1];
				sdata[3 * tid + 2] += sdata[3 * tid + s + 2];
			}
		}
		if (tid < 32) {
			sdata[3 * tid] += sdata[3 * tid + 96];
			sdata[3 * tid + 1] += sdata[3 * tid + 97];
			sdata[3 * tid + 2] += sdata[3 * tid + 98];
			sdata[3 * tid] += sdata[3 * tid + 48];
			sdata[3 * tid + 1] += sdata[3 * tid + 49];
			sdata[3 * tid + 2] += sdata[3 * tid + 50];
			sdata[3 * tid] += sdata[3 * tid + 24];
			sdata[3 * tid + 1] += sdata[3 * tid + 25];
			sdata[3 * tid + 2] += sdata[3 * tid + 26];
			sdata[3 * tid] += sdata[3 * tid + 12];
			sdata[3 * tid + 1] += sdata[3 * tid + 13];
			sdata[3 * tid + 2] += sdata[3 * tid + 14];
			sdata[3 * tid] += sdata[3 * tid + 6];
			sdata[3 * tid + 1] += sdata[3 * tid + 7];
			sdata[3 * tid + 2] += sdata[3 * tid + 8];
			sdata[3 * tid] += sdata[3 * tid + 3];
			sdata[3 * tid + 1] += sdata[3 * tid + 4];
			sdata[3 * tid + 2] += sdata[3 * tid + 5];
		}
		if (tid == 0) {
			outN[blockIdx.x] = sdata[0];
			outD1[blockIdx.x] = sdata[1];
			outD2[blockIdx.x] = sdata[2];
		}
		i += stride;
	}
	//if (blockSize >= 512) {
	//	if (tid < 256) { 
	//		sndata[tid] += sndata[tid + 256];
	//		sd1data[tid] += sd1data[tid + 256];
	//		sd2data[tid] += sd2data[tid + 256];
	//	} __syncthreads();
	//}
	//if (blockSize >= 256) {
	//	if (tid < 128) {
	//		sndata[tid] += sndata[tid + 128];
	//		sd1data[tid] += sd1data[tid + 128];
	//		sd2data[tid] += sd2data[tid + 128];
	//	} __syncthreads();
	//}
	//if (blockSize >= 128) {
	//	if (tid < 64) { 
	//		sndata[tid] += sndata[tid + 64];
	//		sd1data[tid] += sd1data[tid + 64];
	//		sd2data[tid] += sd2data[tid + 64];
	//	} __syncthreads();
	//}
	//if (tid < 32) {
	//	if (blockSize >= 64) { 
	//		sndata[tid] += sndata[tid + 32];
	//		sd1data[tid] += sd1data[tid + 32];
	//		sd2data[tid] += sd2data[tid + 32];
	//	}
	//	if (blockSize >= 32) {
	//		sndata[tid] += sndata[tid + 16];
	//		sd1data[tid] += sd1data[tid + 16];
	//		sd2data[tid] += sd2data[tid + 16];
	//	}
	//	if (blockSize >= 16) {
	//		sndata[tid] += sndata[tid + 8];
	//		sd1data[tid] += sd1data[tid + 8];
	//		sd2data[tid] += sd2data[tid + 8];
	//	}
	//	if (blockSize >= 8) {
	//		sndata[tid] += sndata[tid + 4];
	//		sd1data[tid] += sd1data[tid + 4];
	//		sd2data[tid] += sd2data[tid + 4];
	//	}
	//	if (blockSize >= 4) {
	//		sndata[tid] += sndata[tid + 2];
	//		sd1data[tid] += sd1data[tid + 2];
	//		sd2data[tid] += sd2data[tid + 2];
	//	}
	//	if (blockSize >= 2) {
	//		sndata[tid] += sndata[tid + 1];
	//		sd1data[tid] += sd1data[tid + 1];
	//		sd2data[tid] += sd2data[tid + 1];
	//	}
	//}
}
__global__ void ladKernel(float *a, float *b, float *out, int size) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	int stride = blockDim.x * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < size) {
		sdata[tid] += abs(a[i] - b[i]) + abs(a[i + blockDim.x] - b[i + blockDim.x]);
		i += stride;
		__syncthreads();
	}
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid<s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();

	}
	if (tid < 32) {
		sdata[tid] += sdata[tid + 32];
		__syncthreads();
		sdata[tid] += sdata[tid + 16];
		__syncthreads();
		sdata[tid] += sdata[tid + 8];
		__syncthreads();
		sdata[tid] += sdata[tid + 4];
		__syncthreads();
		sdata[tid] += sdata[tid + 2];
		__syncthreads();
		sdata[tid] += sdata[tid + 1];
		__syncthreads();
	}
	if (tid == 0) {
		out[blockIdx.x] = sdata[0];
	}
}
__global__ void maxKernel(float *array, int size, float* max)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
	int stride = blockDim.x * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < size)
	{
		sdata[tid] = fmaxf(array[i], array[i + blockDim.x]);
		i += stride;
		__syncthreads();

	}

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
		__syncthreads();

	}


	if (tid < 32) {
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
		__syncthreads();
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
		__syncthreads();
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 8]);
		__syncthreads();
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 4]);
		__syncthreads();
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 2]);
		__syncthreads();
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 1]);
		__syncthreads();

	}
	if (tid == 0) {
		max[blockIdx.x] = sdata[0];
	}
}
__global__ void minKernel(float *array, int size, float* min)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
	int stride = blockDim.x * 2 * gridDim.x;
	//sdata[tid] = 0;
	while (i < size)
	{
		sdata[tid] = fminf(array[i], array[i + blockDim.x]);
		i += stride;
		__syncthreads();

	}

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
		__syncthreads();

	}


	if (tid < 32) {
		sdata[tid] = fminf(sdata[tid], sdata[tid + 32]);
		__syncthreads();
		sdata[tid] = fminf(sdata[tid], sdata[tid + 16]);
		__syncthreads();
		sdata[tid] = fminf(sdata[tid], sdata[tid + 8]);
		__syncthreads();
		sdata[tid] = fminf(sdata[tid], sdata[tid + 4]);
		__syncthreads();
		sdata[tid] = fminf(sdata[tid], sdata[tid + 2]);
		__syncthreads();
		sdata[tid] = fminf(sdata[tid], sdata[tid + 1]);
		__syncthreads();

	}
	if (tid == 0) {
		min[blockIdx.x] = sdata[0];
	}
}
__global__ void imrealKernel(cufftComplex *array, int size, float* out)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x  * blockDim.x + threadIdx.x;
	int stride = blockDim.x  * gridDim.x;
	//sdata[tid] = 0;
	while (i < size)
	{
		out[i] = powf(powf(array[i].x, 2.0) + powf(array[i].y, 2.0), 0.5);
		i += stride;
		__syncthreads();

	}
}
__global__ void normalizeKernel(float *array, float min, float max, int size, float* out)
{
	printf("something\n");

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x  * blockDim.x + threadIdx.x;
	int stride = blockDim.x  * gridDim.x;
	while (i < size)
	{
		out[i] = (array[i] - min) / (max - min);
		i += stride;
		__syncthreads();

	}
}

int main(int argc, char *argv[])
{
	string dirPath = "F:\\Downloads\\Converter\\Data\\";
	string samplePath = "F:\\Downloads\\Converter\\samples\\";
	if (argc > 0)
	{
		dirPath = argv[1];
		samplePath = argv[2];
	}
	//float a[644904];
	//for (int i = 0; i < 644904; i++)
	//{
	//	a[i] =  i + 0.5;
	//}
	//printf("max of a: %f\n", findMaxCuda(a, 500));
	compare(dirPath, samplePath);
}

void compare(string dirPath, string samplePath) {
	int sampleSize, audioSize;
	cufftComplex* input;
	cufftComplex* sample;
	vector<string> names = get_all_files_names_within_folder(dirPath);
	vector<string> samplenames = get_all_files_names_within_folder(samplePath);
#pragma omp parallel for
	for (int k = 0; k < samplenames.size(); k++)
	{
		float* ans = new float[names.size()];
		int* sampleArray = readAudio((samplePath + samplenames.at(k)).c_str()  , sampleSize);
		sample = audioFFTCuda(sampleArray, 0, sampleSize);
		float* normS = normalize(sample, sampleSize);
		for (int i = 0; i < names.size(); i++)
		{
			const clock_t begin_time = clock();
			const char* c = (dirPath + names.at(i)).c_str();
			// read audio file
			int* audioArray = readAudio(c, audioSize);
			int sampleCount = (audioSize / sampleSize) * SAMPLERATE;
			float* sims = new float[sampleCount];
			//printf("audio size: %d\n", audioSize);
			//std::cout << "elapsed time: " << float(clock() - begin_time) / CLOCKS_PER_SEC << endl;
			const clock_t begin_time2 = clock();
			if (audioSize < sampleSize) {
				ans[i] = 1000000;
				continue;
			}
			for (int j = 0; j < sampleCount; j++) {
				int start = ((audioSize - sampleSize) / sampleCount) * j;
				int end = start + sampleSize;
				//printf("start: %d\n", start);
				//printf("end: %d\n", end);
				if (end > audioSize)
					break;
				input = audioFFTCuda(audioArray, start, end);
				float* normI = normalize(input, sampleSize);
				sims[j] = ladCuda(normI, normS, sampleSize);
				//sims[j] = cosineSimilarityCuda(normI, normS, sampleSize);
			}
			//std::cout << "elapsed time2: " << float(clock() - begin_time2) / CLOCKS_PER_SEC << endl;
			ans[i] = findMin(sims, sampleCount);
		}
		int minIndex = 0;
		for (int i = 0; i < names.size(); i++)
		{
			//printf("sims: %f %s >> %s\n", ans[i], samplenames.at(k), names.at(i));
			if (ans[minIndex] > ans[i])
				minIndex = i;
		}
		if (ans[minIndex] > 1000) {
			printf("no sample found for %s\n", samplenames.at(k),ans[minIndex]);
		}
		else
			printf("%s >> %s\n",samplenames.at(k), names.at(minIndex),ans[minIndex]);
	}


}

int* readAudio(const char* path, int& size, int start, int end) {
	ifstream file;
	file.open(path);
	if (!file) {
		fprintf(stderr, "Unable to Read the file");
		return 0;
	}
	size = countWords(file);
	int *a = new int[size];
	file.clear();
	file.seekg(0, ios_base::beg);
	int currentSample = 0;
	if (file.is_open()) {
		for (int i = 0; i < size; i++) {
			file >> currentSample;
			a[i] = currentSample;
		}
	}
	return a;

}
int* readAudio(const char* path, int& size) {
	ifstream file;
	file.open(path);
	if (!file) {
		fprintf(stderr, "Unable to Read the file");
		return 0;
	}
	size = countWords(file);
	int *a = new int[size];
	file.clear();
	file.seekg(0, ios_base::beg);
	int currentSample = 0;
	if (file.is_open()) {
		for (int i = 0; i < size; i++) {
			file >> currentSample;
			a[i] = currentSample;
		}
	}
	return a;

}
// Helper function for using CUDA to add vectors in parallel.
cufftComplex* audioFFTCuda(int* array, int start, int end)
{
	int size = end - start;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	cufftComplex *x = (cufftComplex *)malloc(sizeof(cufftComplex)*size);
	cufftComplex* y;

	if (cudaMalloc((void**)&y, sizeof(cufftComplex)*size) != CUFFT_SUCCESS) {
		fprintf(stderr, "Cuda allocation Error. y\n");
		goto Error;
	}

	for (int i = start; i < end; i++) {
		x[i - start].x = array[i];
	}

	if (cudaMemcpy(y, x, sizeof(cufftComplex)*size, cudaMemcpyHostToDevice) != CUFFT_SUCCESS) {
		fprintf(stderr, "Cuda memcpy error");
		goto Error;
	}

	cufftHandle plan;
	if (cufftPlan1d(&plan, size, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Plan Init Failed");
		goto Error;
	}
	if (cufftExecC2C(plan, y, y, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		goto Error;
	}



	// Copy input vectors from host memory to GPU buffers.
	// Launch a kernel on the GPU with one thread for each element.
	//addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	if (cudaMemcpy(x, y, sizeof(cufftComplex)*size, cudaMemcpyDeviceToHost) != CUFFT_SUCCESS) {
		fprintf(stderr, "Cuda Copy to host error");
		goto Error;
	}

	/*for (int i = 0; i < size; i++) {
	printf("T[%d] = %f\n", i, x[i].x);
	}*/


Error:
	cudaFree(y);
	//free(x);
	cufftDestroy(plan);
	/*cudaFree(dev_a);
	cudaFree(dev_b);*/
	return x;
}
float cosineSimilarity(float* a, float* b, int sampleSize) {
	float cosmul = 0;
	float asize = 0;
	float bsize = 0;
	for (int i = 0; i < sampleSize; i++) {
		cosmul += a[i] * b[i];
		asize += pow(a[i], 2);
		bsize += pow(b[i], 2);
	}
	cosmul = abs(cosmul);
	asize = sqrt(asize);
	bsize = sqrt(bsize);
	//printf("cosmul: %f\n", cosmul/(asize * bsize));

	return cosmul / (asize * bsize);

}
float cosineSimilarityCuda(float* a, float* b, int sampleSize) {
	float* dev_a;
	float* dev_b;

	// set device
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, sizeof(float)*sampleSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda allocation Error. dev_a\n");
		printf("cudaMalloc d_a returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, sizeof(float)*sampleSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error");
		printf("cudaCOPY d_a returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, sizeof(float)*sampleSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda allocation Error. dev_b\n");
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, b, sizeof(float)*sampleSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	float* dev_outn;
	float* dev_outd1;
	float* dev_outd2;
	if (cudaMalloc((void**)&dev_outn, sizeof(float)*BLOCK_COUNT) != cudaSuccess) {
		fprintf(stderr, "Cuda allocation Error. 1\n");
		goto Error;
	}
	if (cudaMalloc((void**)&dev_outd1, sizeof(float)*BLOCK_COUNT) != cudaSuccess) {
		fprintf(stderr, "Cuda allocation Error. 2 \n");
		goto Error;
	}
	if (cudaMalloc((void**)&dev_outd2, sizeof(float)*BLOCK_COUNT) != cudaSuccess) {
		fprintf(stderr, "Cuda allocation Error. 3\n");
		goto Error;
	}
	float* outn;
	float* outd1;
	float* outd2;

	outn = (float *)calloc(BLOCK_COUNT, sizeof(float));
	outd1 = (float *)calloc(BLOCK_COUNT, sizeof(float));
	outd2 = (float *)calloc(BLOCK_COUNT, sizeof(float));
	size_t shm_size = 3 * THREAD_COUNT * sizeof(unsigned long long);

	cosineKernel << < BLOCK_COUNT, THREAD_COUNT, shm_size >> > (dev_a, dev_b, dev_outn, dev_outd1, dev_outd2, sampleSize);

	cudaStatus = cudaMemcpy(outd1, dev_outd1, sizeof(float)*BLOCK_COUNT, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outd1 returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMemcpy(outn, dev_outn, sizeof(float)*BLOCK_COUNT, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMemcpy(outd2, dev_outd2, sizeof(float)*BLOCK_COUNT, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outd2 returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	float numerator = 0;
	float denominator1 = 0;
	float denominator2 = 0;
	for (int i = 0; i < BLOCK_COUNT; i++)
	{
		numerator += outn[i];
		denominator1 += outd1[i];
		denominator2 += outd2[i];
	}
	denominator1 = sqrt(denominator1);
	denominator2 = sqrt(denominator2);
	float cossim = numerator / (denominator1 * denominator2);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	return cossim;
}

float ladCuda(float* a, float* b, int sampleSize) {
	float* dev_a;
	float* dev_b;

	// set device
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, sizeof(float)*sampleSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda allocation Error. dev_a\n");
		printf("cudaMalloc d_a returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, sizeof(float)*sampleSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error");
		printf("cudaCOPY d_a returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, sizeof(float)*sampleSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda allocation Error. dev_b\n");
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, b, sizeof(float)*sampleSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	float* dev_out;
	if (cudaMalloc((void**)&dev_out, sizeof(float)*BLOCK_COUNT) != cudaSuccess) {
		fprintf(stderr, "Cuda allocation Error. 1\n");
		goto Error;
	}
	float* out;

	out = (float *)calloc(BLOCK_COUNT, sizeof(float));
	size_t shm_size = THREAD_COUNT * sizeof(unsigned long long);
	ladKernel << < BLOCK_COUNT, THREAD_COUNT, shm_size >> > (dev_a, dev_b, dev_out, sampleSize);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpy(out, dev_out, sizeof(float)*BLOCK_COUNT, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	float ladVal = 0;
	for (int i = 0; i < BLOCK_COUNT; i++)
	{
		//printf("outtt %f", ladVal);
		ladVal += out[i];
	}
Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	return ladVal;
}
float cosineSimilarity(float* a, float* b, int sampleSize, int offset, int audioSize) {
	if (sampleSize + offset > audioSize)
		return 0;
	float cosmul = 0;
	float asize = 0;
	float bsize = 0;
	for (int i = 0; i < sampleSize; i++) {
		cosmul += a[offset + i] * b[i];
		asize += pow(a[offset + i], 2);
		bsize += pow(b[i], 2);
	}
	cosmul = abs(cosmul);
	asize = pow(asize, 0.5);
	bsize = pow(bsize, 0.5);
	//printf("cosmul: %f\n", cosmul/(asize * bsize));

	return cosmul / (asize * bsize);

}
vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	string search_path = folder + "/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

float* normalize(cufftComplex* c, int size) {
	float *a = new float[size];
	float min = findMinC(c, size);
	float max = findMaxC(c, size);
	for (int i = 0; i < size; i++) {
		float temp = sqrt(pow(c[i].x, 2) + pow(c[i].y, 2));
		a[i] = (temp - min) / (max - min);
	}
	return a;
}
float findMax(float* a, int N) {
	float max = a[0];
	for (int i = 1; i < N; i++) {
		if (max < a[i])
			max = a[i];
	}
	return max;
}
float findMaxC(cufftComplex* a, int N) {
	float max = sqrt(pow(a[0].x, 2) + pow(a[0].y, 2));
	for (int i = 1; i < N; i++) {
		float temp = sqrt(pow(a[i].x, 2) + pow(a[i].y, 2));
		if (max < temp)
			max = temp;
	}
	return max;
}
float findMin(float* a, int N) {
	float min = a[0];
	for (int i = 1; i < N; i++) {
		if (min > a[i])
			min = a[i];
	}
	return min;
}
float findMinC(cufftComplex* a, int N) {
	float min = sqrt(pow(a[0].x, 2) + pow(a[0].y, 2));
	for (int i = 1; i < N; i++) {
		float temp = sqrt(pow(a[i].x, 2) + pow(a[i].y, 2));
		if (min > temp)
			min = temp;
	}
	return min;
}
int countWords(std::ifstream& in) {
	int count = 0;
	for (std::string word; in >> word; ++count) {}
	return count;
}
float lad(float* a, float* b, int sampleSize) {
	float size = 0;
	for (int i = 0; i < sampleSize; i++) {
		size += abs(a[i] - b[i]);
	}
	//printf("cosmul: %f\n", cosmul/(asize * bsize));

	return size;

}

float* normalizeCuda(cufftComplex* array, int size) {
	cufftComplex *cufft_array;
	float* hsize_array;
	float* size_array;
	float *h_max;
	float *d_max;
	float* dev_normal;
	float* normal = (float*)malloc(sizeof(float) * size);
	size_t shm_size = THREAD_COUNT * sizeof(unsigned long long);


	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	// allocate memory
	h_max = (float*)malloc(sizeof(float) * BLOCK_COUNT);
	hsize_array = (float*)malloc(sizeof(float) * size);
	cudaStatus = cudaMalloc((void**)&cufft_array, size * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&size_array, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_max, sizeof(float) * BLOCK_COUNT);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	cudaStatus = cudaMemcpy(cufft_array, array, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	// call kernel
	imrealKernel << < BLOCK_COUNT, THREAD_COUNT >> >(cufft_array, size, size_array);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	// copy from device to host
	cudaStatus = cudaMemcpy(hsize_array, size_array, sizeof(float) * size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	float max = findMax(hsize_array, size);
	float min = findMin(hsize_array, size);
	float* dev_min;
	float* dev_max;
	cudaStatus = cudaMalloc((void**)&dev_min, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_max, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_min, &min, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_max, &max, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_normal, sizeof(float) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	float* new_size_array;
	cudaStatus = cudaMalloc((void**)&new_size_array, sizeof(float)*size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMemcpy(new_size_array, size_array, sizeof(float)* size, cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	/*int* dev_size;
	cudaStatus = cudaMalloc((void**)&dev_size, sizeof(int));
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "Cuda memcpy error\n");
	printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
	goto Error;
	}
	printf("aifeu\n");
	int hsize = size;
	cudaStatus = cudaMemcpy(dev_size, &hsize, sizeof(int), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "Cuda memcpy error\n");
	printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
	goto Error;
	}*/
	normalizeKernel << < BLOCK_COUNT, THREAD_COUNT , shm_size>> >(size_array,*dev_min,*dev_max, size, dev_normal);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpy(normal, dev_normal, sizeof(float) * size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}


	/*
	for (int i = 0; i < size; i++)
	{
	printf("imrea: %f", hmax_array[i]);
	}
	*/
	//size_t shm_size = THREAD_COUNT * sizeof(unsigned long long);
	//maxKernel << < BLOCK_COUNT, THREAD_COUNT, shm_size >> >(max_array, size, d_max);

	/*	float max = h_max[0];
	for (int i = 1; i < BLOCK_COUNT; i++) {
	if (max < h_max[i])
	max = h_max[i];
	}
	printf("%faa\n", max);
	*/
	// free memory
Error:
	cudaFree(cufft_array);
	cudaFree(d_max);
	return normal;

}
float findMaxCuda(float* array, int size) {
	float *dev_array;
	float *h_max;
	float *d_max;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	// allocate memory
	h_max = (float*)malloc(sizeof(float) * BLOCK_COUNT);
	cudaStatus = cudaMalloc((void**)&dev_array, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_max, sizeof(float) * BLOCK_COUNT);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_array, array, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	// call kernel
	size_t shm_size = THREAD_COUNT * sizeof(unsigned long long);

	maxKernel << < BLOCK_COUNT, THREAD_COUNT, shm_size >> >(dev_array, size, d_max);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	// copy from device to host
	cudaStatus = cudaMemcpy(h_max, d_max, sizeof(float) * BLOCK_COUNT, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	float max = array[0];
	for (int i = 1; i < size; i++) {
		if (max < array[i])
			max = array[i];
	}
	printf("%faa\n", max);

	// free memory
Error:
	cudaFree(dev_array);
	cudaFree(d_max);
	//free(array);
	//free(h_max);
	return max;

}
float findMinCuda(float* array, int size) {
	float *dev_array;
	float *h_min;
	float *d_min;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	// allocate memory
	h_min = (float*)malloc(sizeof(float) * BLOCK_COUNT);
	cudaStatus = cudaMalloc((void**)&dev_array, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_min, sizeof(float) * BLOCK_COUNT);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_array, array, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}

	// call kernel
	size_t shm_size = THREAD_COUNT * sizeof(unsigned long long);

	minKernel << < BLOCK_COUNT, THREAD_COUNT, shm_size >> >(dev_array, size, d_min);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	// copy from device to host
	cudaStatus = cudaMemcpy(h_min, d_min, sizeof(float) * BLOCK_COUNT, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda memcpy error\n");
		printf("cudaCOPY outn returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		goto Error;
	}
	float minVal = findMin(h_min, BLOCK_COUNT);

	// free memory
Error:
	cudaFree(dev_array);
	cudaFree(d_min);

	return 	minVal;
}

