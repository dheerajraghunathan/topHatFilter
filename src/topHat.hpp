#ifndef TOPHAT_HPP
#define TOPHAT_HPP

#include <vector>
#include <cstdlib>

using namespace std;

// ==============================================================
// ====== Function to generate random values to a 2D array ======
// ==============================================================
void generate_random_data(vector<vector<double>> &array, const int &Nx, const int &Ny) {
    srand(time(nullptr));   // Seed the random number generator

    // Fill the array with random values (0 to 99)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            array[i][j] = rand() % 100;  // Random numbers from 0 to 99
        }
    }

}
// ####################### Done #################################

#endif