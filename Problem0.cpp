#include <iostream>
#include <cstdlib>
using namespace std;

/*
M = total vector length
P = number of processes
p = process rank (0 to P-1)
i = local index (0 to M/P - 1)
*/

int main(int argc, char* argv[]) {

    if(argc != 5) {
        cerr << "Usage: " << argv[0] << " M P p i" << endl;
        return 1;
    }


    int M = atoi(argv[1]); // Total vector length
    int P = atoi(argv[2]); // Number of processes
    int p = atoi(argv[3]); // Process rank
    int i = atoi(argv[4]); // Local index
    
    if(p < 0 || p >= P) {
        cerr << "Error: Process rank p must be between 0 and P-1." << endl;
        return 1;
    }

    if(i < 0 || i >= M/P) {
        cerr << "Error: Local index i must be between 0 and M/P - 1." << endl;
        return 1;
    }
    int base = (M / P); // Base index for process p
    int rem = M % P; // Remainder to distribute among the first rem processes

    int local_index;
    if(p < rem) {
        local_index = p * (base + 1) + i; // Process p gets base+1 elements
    } else {
        local_index = rem * (base + 1) + (p - rem) * base + i; // Process p gets base elements
    }

    int start;
    if(p < rem) {
        start = p * (base + 1); // Starting index for process p
    } else {
        start = rem * (base + 1) + (p - rem) * base; // Starting index for process p
    }

    int i = start + i;

    int p_prime = i % P; // Process rank that owns the global index i
    int i_prime = i / P; // Local index within process p_prime
    cout << "Global index i: " << i << endl;
    cout << "Process rank p': " << p_prime << endl;
    cout << "Local index i': " << i_prime << endl;

    return 0;
   }


