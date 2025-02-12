#include <iostream>

#include "topHat.hpp"

using namespace std;
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        // **************************************************************
        // Declare basic variables
        // **************************************************************
        int Ngx = 8, Ngy = 8;   // Global array size
        int Npx = 2, Npy = 2;   // No. of processors in each direction
        int Nx  = Ngx/Npx;      // Local array size in X direction
        int Ny  = Ngy/Npy;      // Local array size in Y direction
        int numGhost = 2;
		Nx = Nx + numGhost;
		Ny = Ny + numGhost;
		// ####################### Done #################################

        // **************************************************************
	    // Set MPI Parameters
	    // **************************************************************
		// Number of processors
		int procsize; 					
		MPI_Comm_size(MPI_COMM_WORLD, &procsize);

		// Ask MPI to decompose processors in a Cartesian grid
		int dims[2] = {Npx, Npy};
		MPI_Dims_create(procsize, 2, dims);

		// If periodic BC is true then it connects appropriate processors
		int periods[2];
		periods[0] = true;
		periods[1] = true;

		// Let MPI assign arbitrary ranks if it seems necessary
		int reorder = false;

		// Define the communicator
		MPI_Comm comm;
		MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm);

		// My rank in the new communicator
		int myRank;
		MPI_Comm_rank(comm, &myRank);

		// Get my coordinates in the new communicator
		int myCoords[2];
		MPI_Cart_coords(comm, myRank, 2, myCoords);

        // Get global index ranges for each processor
		int ibegin = myCoords[0]*(Nx-numGhost);
		int jbegin = myCoords[1]*(Ny-numGhost);
	    // ####################### Done #################################

        // **************************************************************
        //  Declare Arrays
        // **************************************************************
        Kokkos::View<double**, Kokkos::LayoutRight> globalArray("globalArray", Ngx, Ngy);
		Kokkos::View<double**, Kokkos::LayoutRight> localArray("localArray", Nx, Ny);
        // ####################### Done #################################

        // **************************************************************
        // Generate random values to the global array
        // **************************************************************
        if (myRank==0) generateRandomData(globalArray, Ngx, Ngy);
        // ####################### Done #################################

        // **************************************************************
        // Broadcast the global matrix with random values from rank 0 to all other ranks
        // **************************************************************
        MPI_Bcast(globalArray.data(), Ngx * Ngy, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // ####################### Done #################################
        
        // **************************************************************
        // Print the array
        // **************************************************************
        // Allocate an array in the host space & copy the globalArray
        auto globalArray_output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), globalArray);
        if (myRank==0) {
            cout << "Generated random array:\n";
            for (int i = 0; i < Ngx; i++) {
                for (int j = 0; j < Ngy; j++) {
                    cout << globalArray_output(i,j) << "\t";
                }
                cout << "\n";
            }
        }
        // ####################### Done #################################

        // **************************************************************
        // Divide the global array to subarrays
        // **************************************************************
        Kokkos::parallel_for("copy global values to local arrays",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {Nx-1, Ny-1}),
            KOKKOS_LAMBDA(int i, int j) {
                int ii = ibegin + i - 1;    // -1 because global array has no Ghost nodes
                int jj = jbegin + j - 1;    // -1 because global array has no Ghost nodes
                localArray(i,j)  =  globalArray(ii,jj);
        });
        // ####################### Done #################################
        
        // **************************************************************
        // Halo data transfer - Boundary nodes treated as periodic
        // **************************************************************
        haloTransfer(localArray, Nx, Ny, comm);
        // ####################### Done #################################

        // **************************************************************
        // Apply filtering - Periodic Boundaries
        // **************************************************************
        applyTopHatFilter(localArray, Nx, Ny);
        // ####################### Done #################################
        
        // **************************************************************
        // Print local arrays
        // **************************************************************
        auto localArray_output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localArray);
        Kokkos::deep_copy(localArray_output, localArray);  // Copy updated data from device to host
        Kokkos::fence();
        if (myRank==0) {
            cout << "Local Array - Rank 0:\n";
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    cout << localArray_output(i,j) << "\t";
                }
                cout << "\n";
            }
        }
        if (myRank==1) {
            cout << "Local Array - Rank 1:\n";
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    cout << localArray_output(i,j) << "\t";
                }
                cout << "\n";
            }
        }
        if (myRank==2) {
            cout << "Local Array - Rank 2:\n";
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    cout << localArray_output(i,j) << "\t";
                }
                cout << "\n";
            }
        }
        if (myRank==3) {
            cout << "Local Array - Rank 3:\n";
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    cout << localArray_output(i,j) << "\t";
                }
                cout << "\n";
            }
        }
        // ####################### Done #################################

    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
