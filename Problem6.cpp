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
            cout << "Example: mpirun -np 8 ./problem6 15 4 2" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int M = atoi(argv[1]);  // vector length
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

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    vector<int> x_counts(P);
    vector<int> x_displs(P);

    int base = M / P;
    int rem = M % P;

    for (int r = 0; r < P; r++) {
        x_counts[r] = base + (r < rem ? 1 : 0);
        x_displs[r] = (r == 0) ? 0 : x_displs[r - 1] + x_counts[r - 1];
    }

    int local_x_n = x_counts[row];
    int x_start = x_displs[row];

    vector<double> local_x(local_x_n);

    vector<double> full_x;
    if (world_rank == 0) {
        full_x.resize(M);
        for (int i = 0; i < M; i++)
            full_x[i] = i + 1;  // example values
    }

    if (col == 0) {
        MPI_Scatterv(full_x.data(), x_counts.data(), x_displs.data(), MPI_DOUBLE, local_x.data(), local_x_n, MPI_DOUBLE, 0, col_comm);
    }

    MPI_Bcast(local_x.data(), local_x_n, MPI_DOUBLE, 0, row_comm);

    int local_y_n = (M + Q - 1 - col) / Q;
    vector<double> local_y(local_y_n, 0.0);

    for (int i = 0; i < local_x_n; i++) {
        int global_i = x_start + i;

        int owner_col = global_i % Q;
        int local_idx = global_i / Q;

        if (col == owner_col) {
            local_y[local_idx] = local_x[i];
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, local_y.data(), local_y_n, MPI_DOUBLE, MPI_SUM, col_comm);

    double local_sum = 0.0;

    for (int i = 0; i < local_x_n; i++) {
        int global_i = x_start + i;

        int y_index = global_i / Q;   // from cyclic mapping

        double x_val = local_x[i];
        double y_val = local_y[y_index];

        local_sum += x_val * y_val;
    }
    double global_sum = 0.0;

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (world_rank == 0) {
        cout << "Dot product = " << global_sum << endl;
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}