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
            cout << "Example: mpirun -np 8 ./problem3 15 4 2" << endl;
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

    int row = world_rank / Q;
    int col = world_rank % Q;

    MPI_Comm row_comm;
    MPI_Comm col_comm;

    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    vector<int> x_counts(P);
    vector<int> x_displs(P);

    int base = M / P;
    int rem = M % P;

    for (int r = 0; r < P; r++) {
        x_counts[r] = base + (r < rem ? 1 : 0);

        if (r == 0)
            x_displs[r] = 0;
        else
            x_displs[r] = x_displs[r - 1] + x_counts[r - 1];
    }

    int local_x_n = x_counts[row];
    int x_start = x_displs[row];

    vector<double> local_x(local_x_n);

    vector<double> full_x;
    if (world_rank == 0) {
        full_x.resize(M);
        for (int i = 0; i < M; i++) {
            full_x[i] = i;
        }
    }

    // Scatter x down column 0
    if (col == 0) {
        MPI_Scatterv(
            full_x.data(),
            x_counts.data(),
            x_displs.data(),
            MPI_DOUBLE,
            local_x.data(),
            local_x_n,
            MPI_DOUBLE,
            0,
            col_comm
        );
    }

    MPI_Bcast(
        local_x.data(),
        local_x_n,
        MPI_DOUBLE,
        0,
        row_comm
    );


    int local_y_n = (M + Q - 1 - col) / Q;
    vector<double> local_y(local_y_n, -1.0);

    // Each process owns a chunk of x.
    // For every global x index J in that chunk,
    // place it into y if this column owns J.
    for (int k = 0; k < local_x_n; k++) {
        int J = x_start + k;

        int owner_col = J % Q;
        int local_j = J / Q;

        if (col == owner_col) {
            local_y[local_j] = local_x[k];
        }
    }


    for (int j = 0; j < local_y_n; j++) {
        if (local_y[j] < 0) {
            local_y[j] = 0.0;
        }
    }

    MPI_Allreduce(
        MPI_IN_PLACE,
        local_y.data(),
        local_y_n,
        MPI_DOUBLE,
        MPI_SUM,
        col_comm
    );

    // Print in rank order
    for (int r = 0; r < world_size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == r) {
            cout << "World rank " << world_rank
                 << " grid(" << row << "," << col << ")" << endl;

            cout << "  local_x = ";
            for (int i = 0; i < local_x_n; i++) {
                cout << local_x[i] << " ";
            }
            cout << endl;

            cout << "  local_y scatter = ";
            for (int j = 0; j < local_y_n; j++) {
                int global_J = j * Q + col;
                if (global_J < M) {
                    cout << "y[" << global_J << "]=" << local_y[j] << " ";
                }
            }
            cout << endl << endl;
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}