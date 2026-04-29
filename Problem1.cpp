#include <iostream>
#include <cstdlib>
#include <mpi.h>
using namespace std;


int main(int argc, char* argv[]) {

    if(argc != 3) {
        cerr << "Usage: " << argv[0] << " M" << endl;
        return 1;
    }

    int P = atoi(argv[1]);
    int Q = atoi(argv[2]);

    int world_size, world_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(P < 1 || Q < 1) {
        cerr << "Error: P and Q must be positive integers." << endl;
        MPI_Finalize();
        return 1;
    }

    if(world_size != P * Q) {
        cerr << "Error: The number of processes must be equal to P * Q." << endl;
        MPI_Finalize();
        return 1;
    }

    //First split

    int color1 = world_rank / Q; // Color for the first split
    MPI_Comm row_comm; // Communicator for the first split
    MPI_Comm_split(MPI_COMM_WORLD, color1, world_rank, &row_comm);
    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    int local_value = row_rank; // Each process in the same row gets the same value
    int sum_result = 0;

    MPI_Reduce(&local_value, &sum_result, 1, MPI_INT, MPI_SUM, 0, row_comm);

    if(row_rank == 0) {
        cout << "Row " << color1 << " sum: " << sum_result << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //Second split
    int color2 = world_rank % Q; // Color for the second split
    MPI_Comm col_comm; // Communicator for the second split
    MPI_Comm_split(MPI_COMM_WORLD, color2, world_rank, &col_comm);
    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    int broadcast_value = 0;
    if(col_rank == 0) {
        broadcast_value = world_rank; // Each process in the same column gets the same value
    }

    MPI_Bcast(&broadcast_value, 1, MPI_INT, 0, col_comm);

    cout << "Process " << world_rank << " in column " << color2 << " received broadcast value: " << broadcast_value << endl;

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    
   
    MPI_Finalize();
    return 0;
}



