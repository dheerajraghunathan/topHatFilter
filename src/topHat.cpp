#include <iostream>

#include "topHat.hpp"

using namespace std;
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // **************************************************************
        // Declare basic variables
        // **************************************************************
        int Nx = 8, Ny = 8;  // Global array size
        // ####################### Done #################################

        // **************************************************************
        //  Declare Arrays
        // **************************************************************
        // Allocate memory for solution variables
		Kokkos::View<double**, Kokkos::LayoutRight> globalArray("globalArray", Nx, Ny); //Layout right for MPI data transfer
        Kokkos::View<double**, Kokkos::LayoutRight> globalArray1("globalArray1", Nx, Ny); //Layout right for MPI data transfer
        Kokkos::View<double**, Kokkos::LayoutRight> globalArray2("globalArray2", Nx, Ny); //Layout right for MPI data transfer
        // ####################### Done #################################

        // **************************************************************
        // Generate random values to the global array
        // **************************************************************
        generateRandomData(globalArray, Nx, Ny);
        // ####################### Done #################################
        
        // **************************************************************
        // Print the array
        // **************************************************************
        // Allocate an array in the host space & copy the globalArray
        auto globalArray_output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), globalArray);
        cout << "Generated random array:\n";
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                cout << globalArray_output(i,j) << "\t";
            }
            cout << "\n";
        }
        // ####################### Done #################################

        // **************************************************************
        // Declare extra arrays to test different BCs
        // **************************************************************
        Kokkos::deep_copy(globalArray1, globalArray);
        Kokkos::deep_copy(globalArray2, globalArray);
        // ####################### Done #################################

        // **************************************************************
        // Apply filtering
        // **************************************************************
        applyTopHatFilter(globalArray, Nx, Ny);
        Kokkos::deep_copy(globalArray_output, globalArray);  // Copy updated data from device to host
        // ####################### Done #################################

        // **************************************************************
        // Print the array
        // **************************************************************
        cout << "Filtered array - No BC:\n";
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                cout << globalArray_output(i,j) << "\t";
            }
            cout << "\n";
        }
        // ####################### Done #################################

        // **************************************************************
        // Apply filtering - Periodic BC
        // **************************************************************
        applyTopHatFilterPeriodicBC(globalArray1, Nx, Ny);
        // ####################### Done #################################

        // **************************************************************
        // Print the array
        // **************************************************************
        Kokkos::deep_copy(globalArray_output, globalArray1);  // Copy updated data from device to host
        cout << "Filtered array - Periodic BC:\n";
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                cout << globalArray_output(i,j) << "\t";
            }
            cout << "\n";
        }
        // ####################### Done #################################

        // **************************************************************
        // Apply filtering - Zero BC
        // **************************************************************
        applyTopHatFilterZeroBC(globalArray2, Nx, Ny);
        // ####################### Done #################################

        // **************************************************************
        // Print the array
        // **************************************************************
        Kokkos::deep_copy(globalArray_output, globalArray2);  // Copy updated data from device to host
        cout << "Filtered array - Zero BC:\n";
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                std::cout << globalArray_output(i,j) << "\t";
            }
            std::cout << "\n";
        }
        // ####################### Done #################################
    }
    Kokkos::finalize();
    return 0;
}
