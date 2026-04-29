#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 4) {
        if (world_rank == 0) {
            cout << "Usage: " << argv[0] << " M P Q" << endl;
            cout << "Example: mpirun -np 8 ./problem2 15 4 2" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int M = atoi(argv[1]);
    int P = atoi(argv[2]);
    int Q = atoi(argv[3]);

    if (world_size != P * Q) {
        if (world_rank == 0) {
            cout << "Error: world_size must equal P * Q" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Convert world rank into 2D grid coordinates
    int row = world_rank / Q;
    int col = world_rank % Q;

    // Create row communicator: same row
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);

    // Create column communicator: same column
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    // Compute linear load-balanced chunk sizes over P rows
    vector<int> counts(P);
    vector<int> displs(P);

    int base = M / P;
    int rem = M % P;

    for (int r = 0; r < P; r++) {
        counts[r] = base + (r < rem ? 1 : 0);

        if (r == 0) {
            displs[r] = 0;
        } else {
            displs[r] = displs[r - 1] + counts[r - 1];
        }
    }

    int local_n = counts[row];

    vector<double> local_x(local_n);

    // Full x only exists initially on process (0,0), which is world rank 0
    vector<double> full_x;

    if (world_rank == 0) {
        full_x.resize(M);

        for (int i = 0; i < M; i++) {
            full_x[i] = i;
        }
    }


    if (col == 0) {
        MPI_Scatterv(
            full_x.data(),
            counts.data(),
            displs.data(),
            MPI_DOUBLE,
            local_x.data(),
            local_n,
            MPI_DOUBLE,
            0,
            col_comm
        );
    }


    MPI_Bcast(
        local_x.data(),
        local_n,
        MPI_DOUBLE,
        0,
        row_comm
    );


    vector<double> y(M);

    MPI_Allgatherv(
        local_x.data(),
        local_n,
        MPI_DOUBLE,
        y.data(),
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        col_comm
    );

    // Print results in rank order
    for (int r = 0; r < world_size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == r) {
            cout << "World rank " << world_rank
                 << " at grid position (" << row << "," << col << ")"
                 << endl;

            cout << "  local_x = ";
            for (int i = 0; i < local_n; i++) {
                cout << local_x[i] << " ";
            }
            cout << endl;

            cout << "  y = ";
            for (int i = 0; i < M; i++) {
                cout << y[i] << " ";
            }
            cout << endl << endl;
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}
