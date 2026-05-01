#include <stdio.h>

//host to device function to do row wise addition of vector and matrix
__global__ void additionKernel(int *vector, int*matrix){
    int thread = threadIdx.x;
    if(thread == 0 || thread == 1 || thread == 2){matrix[thread] += vector[0];}
    if(thread == 3 || thread == 4 || thread == 5){matrix[thread] += vector[1];}
    if(thread == 6 || thread == 7 || thread == 8){matrix[thread] += vector[2];}
}


int main(void) {

    int vector[3] = {221, 12, 157};
    int matrix[3][3] = {
        {130, 147, 115},
        {224, 158, 187},
        {54, 158, 120}
    };
    printf("pre op matrix\n");
    //print pre op matrix
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }

    int vector_size = sizeof(vector);
    int matrix_size = sizeof(matrix);
    
    //gpu/device copies
    int *d_vector;
    int *d_matrix;
    cudaMalloc((void**)&d_matrix, matrix_size);
    cudaMalloc((void**)&d_vector, vector_size);
    
    //copy cpu memory onto device copies
    cudaMemcpy(d_matrix, matrix, matrix_size,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_vector, vector, vector_size, cudaMemcpyHostToDevice);

    additionKernel<<<1,9>>>(d_vector, d_matrix);

    cudaMemcpy(matrix, d_matrix, matrix_size, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_vector);


    printf("post op matrix\n");
    //print pre op matrix
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
	return 0;
}
