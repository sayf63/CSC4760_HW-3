#include <iostream>
#include <iomanip>
using namespace std;

int main() {
    const int N = 100000;
    const double tiny = 1e-18;

    // Case 1: start at 1.0
    double a = 1.0;
    for(int i = 0; i < N; i++) {
        a += tiny;
    }

    // Case 2: accumulate first, then add 1
    double b = 0.0;
    for(int i = 0; i < N; i++) {
        b += tiny;
    }
    b += 1.0;

    // Print high precision
    cout << setprecision(17);

    cout << "a = " << a << endl;
    cout << "b = " << b << endl;

    double diff = a - b;
    cout << "a - b = " << diff << endl;

    // Print hex representations
    cout << hexfloat;
    cout << endl << "Hex representations:" << endl;
    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    cout << "diff = " << diff << endl;

    return 0;
}