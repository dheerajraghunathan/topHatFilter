#ifndef TOPHAT_HPP
#define TOPHAT_HPP

#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip> 
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <mpi.h>

using namespace std;

// ==============================================================
// ====== Function to generate random values to a 2D array ======
// ==============================================================
void generateRandomData(Kokkos::View<double**, Kokkos::LayoutRight> globalArray, const int &Nx, const int &Ny) {
    // Create random number pool
    Kokkos::Random_XorShift64_Pool<> rand_pool(time(NULL)); ///*seed=*/12345

    // Parallel random number generation
    Kokkos::parallel_for("GenerateRandom2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int i, int j) {
            auto rand_gen = rand_pool.get_state();
            globalArray(i, j) = rand_gen.drand(0.0, 100.0);  // Random double in [min_val, max_val)
            rand_pool.free_state(rand_gen);
    });
    Kokkos::fence(); 
}
// ####################### Done #################################

// ==============================================================
// ======= Function to display and write the output array =======
// ==============================================================
void printArray(Kokkos::View<double**, Kokkos::LayoutRight> globalArray, const int &Ngx, const int &Ngy, 
                const int &myRank, std::string filename ) {
    // Allocate an array in the host space & copy the globalArray
    auto globalArray_output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), globalArray);
    if (myRank==0) {
        if (filename=="randomArray")        cout << "Generated random array:\n";
        else if (filename=="filteredArray") cout << "Filtered array:\n";
        else                                cout << "Incorrect filename.\n";

        cout << fixed << setprecision(2);   // All values printed with two decimal places

        // Print to screen
        for (int j = 0; j < Ngy; j++) {
            for (int i = 0; i < Ngx; i++) {
                cout << globalArray_output(i,j) << "\t";
            }
            cout << "\n";
        }
        
        // Write a file
        ofstream outputFile(filename+ ".txt");
        if (outputFile.is_open()) {
            outputFile << std::fixed << std::setprecision(2);  // Set to 2 decimal places
            for (int j = 0; j < Ngy; j++) {
                for (int i = 0; i < Ngx; i++) {
                    outputFile << globalArray_output(i, j) << "\t";  // Tab-separated values
                }
                outputFile << "\n";  // Newline after each row
            }
            outputFile.close();
        } 
        else  
            cerr << "Error in writing output file\n";
    }
}
// ####################### Done #################################

// ==============================================================
// ======= Function to apply top-hat filter to a 2D array =======
// ==============================================================
void applyTopHatFilter(Kokkos::View<double**, Kokkos::LayoutRight> localArray, const int &Nx, const int &Ny) {
    Kokkos::View<double**, Kokkos::LayoutRight> tempArray("tempArray", Nx, Ny);
    Kokkos::deep_copy(tempArray, localArray);          // Copying to a temp array
    const double inv9 = 1.0/9.0;

    Kokkos::parallel_for("TopHatFilter",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {Nx-1, Ny-1}),
        KOKKOS_LAMBDA(int i, int j) {
            localArray(i,j)  =  inv9 * (tempArray(i,j) + tempArray(i+1,j)   + tempArray(i-1,j)   +
                                                         tempArray(i,j+1)   + tempArray(i,j-1)   +
                                                         tempArray(i-1,j-1) + tempArray(i-1,j+1) +
                                                         tempArray(i+1,j-1) + tempArray(i+1,j+1) ) ;
    });

}
// ####################### Done #################################

// ==============================================================
// ====== Halo data transfer between different processors =======
// ==============================================================
void haloTransfer( Kokkos::View<double**, Kokkos::LayoutRight> localArray, 
                   const int &Nx, const int &Ny, const MPI_Comm& comm ) {
    int myRank;
	MPI_Comm_rank(comm, &myRank);

    // Find neighbours in the new communicator
    int neigborProc[4];
    enum DIRECTIONS {LEFT, RIGHT, DOWN, UP};
    // Shift tells us our neighbours
    MPI_Cart_shift(comm, 0, 1, &neigborProc[LEFT], &neigborProc[RIGHT]);
    MPI_Cart_shift(comm, 1, 1, &neigborProc[DOWN], &neigborProc[UP]);
            
    MPI_Status status;
    MPI_Request request[8] = {MPI_REQUEST_NULL};

    // ==========  Data tranfer ======
    //i = 1  	 ---->>>>  	i = Nx-1
    //i = Nx-2   ---->>>>  	i = 0
    //j = 1  	 ---->>>>  	j = Ny-1
    //j = Ny-2   ---->>>>  	j = 0
    // ===============================

    // STEP1 : Sending & receiving information - Column wise (i=1 to i=Nx-1)
    auto sendBuffer1 = Kokkos::subview(localArray, 1, Kokkos::ALL() );
    auto recvBuffer1 = Kokkos::subview(localArray, Nx-1, Kokkos::ALL() );
    if (neigborProc[0] != MPI_PROC_NULL) {
        Kokkos::fence();  // Ensure data readiness
        MPI_Isend(sendBuffer1.data(), Ny, MPI_DOUBLE, neigborProc[0], myRank,   comm, &request[0]);
    }
    if (neigborProc[1] != MPI_PROC_NULL) {
        MPI_Irecv(recvBuffer1.data(), Ny, MPI_DOUBLE, neigborProc[1], neigborProc[1],   comm,  &request[1]);
    }
    MPI_Wait(&request[0], &status); 
    MPI_Wait(&request[1], &status); 
    
    // STEP2 : Sending & receiving information - Column wise (i=Nx-2 to i=0)
    auto sendBuffer2 = Kokkos::subview(localArray, Nx-2, Kokkos::ALL() );
    auto recvBuffer2 = Kokkos::subview(localArray, 0, Kokkos::ALL() );
    if (neigborProc[1] != MPI_PROC_NULL) {
        Kokkos::fence();  // Ensure data readiness
        MPI_Isend(sendBuffer2.data(), Ny, MPI_DOUBLE, neigborProc[1], myRank,   comm, &request[2]);
    }
    if (neigborProc[0] != MPI_PROC_NULL) {
        MPI_Irecv(recvBuffer2.data(), Ny, MPI_DOUBLE, neigborProc[0], neigborProc[0],   comm,  &request[3]);
    }
    MPI_Wait(&request[2], &status); 
    MPI_Wait(&request[3], &status); 

    // =========================================================================================
    /* For STEP1 and STEP2, subviews yield contiguous memory while using Kokkos::LayoutRight. 
       For STEP3 and STEP4, subviews yield non-contiguous memory & hence send/recv require
       packing the data into a contiguous buffer before MPI data transfer. */
    // =========================================================================================
    
    // STEP3 : Sending & receiving information - Row wise (j=1 to j=Ny-1)
    Kokkos::View<double*> sendBuffer3("sendBuffer", Nx);
    Kokkos::View<double*> recvBuffer3("recvBuffer", Nx);
    if (neigborProc[2] != MPI_PROC_NULL) {
        Kokkos::parallel_for(Nx, KOKKOS_LAMBDA(int i) {
            sendBuffer3(i) = localArray(i,1);
        });
        Kokkos::fence();
        MPI_Isend(sendBuffer3.data(), Nx, MPI_DOUBLE, neigborProc[2], myRank, comm, &request[4]);
    }
    if (neigborProc[3] != MPI_PROC_NULL) {
        MPI_Irecv(recvBuffer3.data(), Nx, MPI_DOUBLE, neigborProc[3], neigborProc[3], comm, &request[5]);
        MPI_Wait(&request[5], &status); 
        Kokkos::parallel_for(Nx, KOKKOS_LAMBDA(int i) {
            localArray(i,Ny-1) = recvBuffer3(i);
        });
        Kokkos::fence();
    }
    MPI_Wait(&request[4], &status); 
    
    // STEP4 : Sending & receiving information - Row wise (j=Ny-2 to j=0)
    Kokkos::View<double*> sendBuffer4("sendBuffer", Nx);
    Kokkos::View<double*> recvBuffer4("recvBuffer", Nx);
    if (neigborProc[3] != MPI_PROC_NULL) {
        Kokkos::parallel_for(Nx, KOKKOS_LAMBDA(int i) {
            sendBuffer4(i) = localArray(i,Ny-2);
        });
        Kokkos::fence();
        MPI_Isend(sendBuffer4.data(), Nx, MPI_DOUBLE, neigborProc[3], myRank, comm, &request[6]);
    }
    if (neigborProc[2] != MPI_PROC_NULL) {
        MPI_Irecv(recvBuffer4.data(), Nx, MPI_DOUBLE, neigborProc[2], neigborProc[2],   comm,  &request[7]);
        MPI_Wait(&request[7], &status); 
        Kokkos::parallel_for(Nx, KOKKOS_LAMBDA(int i) {
            localArray(i,0) = recvBuffer4(i);
        });
        Kokkos::fence();
    }
    MPI_Wait(&request[6], &status);
}
// ####################### Done #################################


// ==============================================================
// ======== Gather all subarrays data to the global array =======
// ==============================================================
void gatherData( Kokkos::View<double**, Kokkos::LayoutRight> globalArray, 
                 Kokkos::View<double**, Kokkos::LayoutRight> localArray, 
                 const int &Nx,  const int &Ny, 
                 const int &Npx, const int &Npy, 
                 const int &numGhost, const MPI_Comm& comm ) {
    // MPI essentials
    int myRank, ibegin, jbegin, myCoords[2];
    MPI_Comm_rank(comm, &myRank);
    MPI_Cart_coords(comm, myRank, 2, myCoords);

    // Declare a flattened array (contiguous memory) for data transfer
    Kokkos::View<double*> buffer("buffer", (Nx-numGhost)*(Ny-numGhost));

    MPI_Request request[4];
    MPI_Status status;

    if (myRank == 0) {
        // Copy rank 0's own data directly
        Kokkos::parallel_for("copy local to global in rank 0",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {Nx-1, Ny-1}),
            KOKKOS_LAMBDA(int i, int j) {
                globalArray(i-1,j-1) = localArray(i,j);
        });
        Kokkos::fence();

        // Receive data from other ranks
        for (int proc = 1; proc < Npx*Npy; proc++) {
            // We are in rank 0. So we require myCoords of the source rank
            MPI_Irecv(myCoords, 2, MPI_INT, proc, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Wait(&request[0], &status); 

            // Receiving the flattened array other ranks
            MPI_Irecv(buffer.data(), (Nx-numGhost)*(Ny-numGhost), MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, &request[1]);
            MPI_Wait(&request[1], &status);
    
            ibegin = myCoords[0]*(Nx-numGhost);     // Helps to find the global coordinates
            jbegin = myCoords[1]*(Ny-numGhost);     // Helps to find the global coordinates

            Kokkos::parallel_for("copy local arrays back to global array",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx-numGhost, Ny-numGhost}),
                KOKKOS_LAMBDA(int i, int j) {
                    int index = i + j*(Ny-numGhost); // Index of flattened array
                    int ii = ibegin + i;    // Global coordinates
                    int jj = jbegin + j;    // Global coordinates
                    globalArray(ii,jj) = buffer(index);
            });
            Kokkos::fence();
        }
    }
    else {
        Kokkos::parallel_for("copy local arrays back to global array",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx-numGhost, Ny-numGhost}),
            KOKKOS_LAMBDA(int i, int j) {
                int index = i + j*(Ny-numGhost); // Index of flattened array
                buffer(index) = localArray(i+1,j+1);
        });
        Kokkos::fence();

        // Sending myCoords is required to get index of global arrays
        MPI_Isend(myCoords, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &request[2]);
        MPI_Isend(buffer.data(), (Nx-numGhost)*(Ny-numGhost), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request[3]);

        // Send local data to rank 0
        MPI_Wait(&request[2], &status); 
        MPI_Wait(&request[3], &status); 
    }

}
// ####################### Done #################################


#endif