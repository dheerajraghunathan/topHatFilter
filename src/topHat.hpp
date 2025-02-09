#ifndef TOPHAT_HPP
#define TOPHAT_HPP

#include <vector>
#include <cstdlib>

using namespace std;

// ==============================================================
// ====== Function to generate random values to a 2D array ======
// ==============================================================
void generateRandomData(vector<vector<double>> &array, const int &Nx, const int &Ny) {
    srand(time(nullptr));   // Seed the random number generator

    // Fill the array with random values (0 to 99)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            array[i][j] = rand() % 100;  // Random numbers from 0 to 99
        }
    }

}
// ####################### Done #################################

// ==============================================================
// ======= Function to apply top-hat filter to a 2D array =======
// ==============================================================
void applyTopHatFilter(vector<vector<double>> &array, const int &Nx, const int &Ny) {
    vector<vector<double>> tempArray = array;      // Copying to a temp array
    const double inv9 = 1.0/9.0;
    for (int i = 1; i < Nx-1; i++) {
        for (int j = 1; j < Ny-1; j++) {
            array[i][j]  =  inv9 * (tempArray[i][j] + tempArray[i+1][j]   + tempArray[i-1][j]   +
                                                      tempArray[i][j+1]   + tempArray[i][j-1]   +
                                                      tempArray[i-1][j-1] + tempArray[i-1][j+1] +
                                                      tempArray[i+1][j-1] + tempArray[i+1][j+1] ) ;
        }
    }

}
// ####################### Done #################################

// ==============================================================
// ===== Function to apply top-hat filter with Periodic BC ======
// ==============================================================
void applyTopHatFilterPeriodicBC(vector<vector<double>> &array, const int &Nx, const int &Ny) {
    vector<vector<double>> tempArray = array;      // Copying to a temp array
    const double inv9 = 1.0/9.0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double i_east  = i+1;
            double i_west  = i-1;
            double j_north = j+1;
            double j_south = j-1;

            if (i==0)    i_west  = Nx-1;
            if (i==Nx-1) i_east  = 0;
            if (j==0)    j_south = Ny-1;
            if (j==Ny-1) j_north = 0;
 
            double centre    = tempArray[i][j];
            double east      = tempArray[i_east][j];
            double west      = tempArray[i_west][j];
            double north     = tempArray[i][j_north];
            double south     = tempArray[i][j_south];
            double northeast = tempArray[i_east][j_north];
            double northwest = tempArray[i_west][j_north];
            double southeast = tempArray[i_east][j_south];
            double southwest = tempArray[i_west][j_south];

            array[i][j]  =  inv9 * ( centre + east + west + north + south +
                                     northeast + northwest + southeast + southwest ) ;
        }
    }

}
// ####################### Done #################################

// ==============================================================
// ======= Function to apply top-hat filter with Zero BC  =======
// ==============================================================
void applyTopHatFilterZeroBC(vector<vector<double>> &array, const int &Nx, const int &Ny) {
    vector<vector<double>> tempArray(Nx+2, vector<double>(Ny+2, 0)); // Temp array initialised to zero
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            tempArray[i+1][j+1] = array[i][j]; // Offset by 1 to create zero borders
        }
    }
    const double inv9 = 1.0/9.0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            // Indices of tempArray
            int ii = i+1;  
            int jj = j+1;

            double i_east  = ii+1;
            double i_west  = ii-1;
            double j_north = jj+1;
            double j_south = jj-1;
            
            double centre    = tempArray[ii][jj];
            double east      = tempArray[i_east][jj];
            double west      = tempArray[i_west][jj];
            double north     = tempArray[ii][j_north];
            double south     = tempArray[ii][j_south];
            double northeast = tempArray[i_east][j_north];
            double northwest = tempArray[i_west][j_north];
            double southeast = tempArray[i_east][j_south];
            double southwest = tempArray[i_west][j_south];

            array[i][j]  =  inv9 * ( centre + east + west + north + south +
                                     northeast + northwest + southeast + southwest ) ;
        }
    }

}
// ####################### Done #################################

#endif