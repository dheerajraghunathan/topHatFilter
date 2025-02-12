#ifndef TOPHAT_HPP
#define TOPHAT_HPP

#include <vector>
#include <cstdlib>
#include <ctime>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <mpi.h>

using namespace std;

// ==============================================================
// ====== Function to generate random values to a 2D array ======
// ==============================================================
void generateRandomData(Kokkos::View<double**, Kokkos::LayoutRight> array, const int &Nx, const int &Ny) {
    // Create random number pool
    Kokkos::Random_XorShift64_Pool<> rand_pool(time(NULL)); ///*seed=*/12345

    // Parallel random number generation
    Kokkos::parallel_for("GenerateRandom2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int i, int j) {
            auto rand_gen = rand_pool.get_state();
            array(i, j) = rand_gen.drand(0.0, 100.0);  // Random double in [min_val, max_val)
            rand_pool.free_state(rand_gen);
    });
}
// ####################### Done #################################

// ==============================================================
// ======= Function to apply top-hat filter to a 2D array =======
// ==============================================================
void applyTopHatFilter(Kokkos::View<double**, Kokkos::LayoutRight> array, const int &Nx, const int &Ny) {
    Kokkos::View<double**, Kokkos::LayoutRight> tempArray("tempArray", Nx, Ny);
    Kokkos::deep_copy(tempArray, array);          // Copying to a temp array
    const double inv9 = 1.0/9.0;

    Kokkos::parallel_for("TopHatFilter",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {Nx-1, Ny-1}),
        KOKKOS_LAMBDA(int i, int j) {
            array(i,j)  =  inv9 * (tempArray(i,j) + tempArray(i+1,j)   + tempArray(i-1,j)   +
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
    // Let consider dims[0] = X, so the shift tells us our left and right neighbours
    MPI_Cart_shift(comm, 0, 1, &neigborProc[LEFT], &neigborProc[RIGHT]);
    // Let consider dims[1] = Y, so the shift tells us our up and down neighbours
    MPI_Cart_shift(comm, 1, 1, &neigborProc[DOWN], &neigborProc[UP]);
            
    MPI_Status status;
    MPI_Request request[8] = {MPI_REQUEST_NULL};

    // ==========  Data tranfer ======
    //i = 1  	 ---->>>>  	i = Nx-1
    //i = Nx-2   ---->>>>  	i = 0
    //j = 1  	 ---->>>>  	j = Ny-1
    //j = Ny-2   ---->>>>  	j = 0

    // STEP1 : Sending & receiving information - Column wise (i=1 to i=Nx-1)
    auto sendBuffer1 = Kokkos::subview(localArray, 1, Kokkos::ALL() );
    auto recvBuffer1 = Kokkos::subview(localArray, Nx-1, Kokkos::ALL() );
    if (neigborProc[0] != MPI_PROC_NULL)
    {
        Kokkos::fence();  // Ensure data readiness
        MPI_Isend(sendBuffer1.data(), Ny, MPI_DOUBLE, neigborProc[0], myRank,   comm, &request[0]);
    }
    if (neigborProc[1] != MPI_PROC_NULL)
    {
        MPI_Irecv(recvBuffer1.data(), Ny, MPI_DOUBLE, neigborProc[1], neigborProc[1],   comm,  &request[1]);
    }
    MPI_Wait(&request[0], &status); //blocks and waits for destination process to receive data
    MPI_Wait(&request[1], &status); //blocks and waits for destination process to receive data
    
    // STEP2 : Sending & receiving information - Column wise (i=Nx-2 to i=0)
    auto sendBuffer2 = Kokkos::subview(localArray, Nx-2, Kokkos::ALL() );
    auto recvBuffer2 = Kokkos::subview(localArray, 0, Kokkos::ALL() );
    if (neigborProc[1] != MPI_PROC_NULL)
    {
        Kokkos::fence();  // Ensure data readiness
        MPI_Isend(sendBuffer2.data(), Ny, MPI_DOUBLE, neigborProc[1], myRank,   comm, &request[2]);
    }
    if (neigborProc[0] != MPI_PROC_NULL)
    {
        MPI_Irecv(recvBuffer2.data(), Ny, MPI_DOUBLE, neigborProc[0], neigborProc[0],   comm,  &request[3]);
    }
    MPI_Wait(&request[2], &status); //blocks and waits for destination process to receive data
    MPI_Wait(&request[3], &status); //blocks and waits for destination process to receive data

    // =========================================================================================
    /* For STEP1 and STEP2, subviews yield contiguous memory while using Kokkos::LayoutRight. 
       For STEP3 and STEP4, subviews yield non-contiguous memory & hence send/recv require
       packing the data into a contiguous buffer before MPI data transfer. */
    // =========================================================================================
    
    // STEP3 : Sending & receiving information - Row wise (j=1 to j=Ny-1)
    Kokkos::View<double*> sendBuffer3("sendBuffer", Nx);
    Kokkos::View<double*> recvBuffer3("recvBuffer", Nx);
    if (neigborProc[2] != MPI_PROC_NULL)
    {
        Kokkos::parallel_for(Nx, KOKKOS_LAMBDA(int i) {
            sendBuffer3(i) = localArray(i,1);
        });
        Kokkos::fence();
        MPI_Isend(sendBuffer3.data(), Nx, MPI_DOUBLE, neigborProc[2], myRank, comm, &request[4]);
    }
    if (neigborProc[3] != MPI_PROC_NULL)
    {
        MPI_Irecv(recvBuffer3.data(), Nx, MPI_DOUBLE, neigborProc[3], neigborProc[3], comm, &request[5]);
        MPI_Wait(&request[5], &status); //blocks and waits for destination process to receive data
        Kokkos::parallel_for(Nx, KOKKOS_LAMBDA(int i) {
            localArray(i,Ny-1) = recvBuffer3(i);
        });
        Kokkos::fence();
    }
    MPI_Wait(&request[4], &status); //blocks and waits for destination process to receive data
    
    // STEP4 : Sending & receiving information - Row wise (j=Ny-2 to j=0)
    Kokkos::View<double*> sendBuffer4("sendBuffer", Nx);
    Kokkos::View<double*> recvBuffer4("recvBuffer", Nx);
    if (neigborProc[3] != MPI_PROC_NULL)
    {
        Kokkos::parallel_for(Nx, KOKKOS_LAMBDA(int i) {
            sendBuffer4(i) = localArray(i,Ny-2);
        });
        Kokkos::fence();
        MPI_Isend(sendBuffer4.data(), Nx, MPI_DOUBLE, neigborProc[3], myRank, comm, &request[6]);
    }
    if (neigborProc[2] != MPI_PROC_NULL)
    {
        MPI_Irecv(recvBuffer4.data(), Nx, MPI_DOUBLE, neigborProc[2], neigborProc[2],   comm,  &request[7]);
        MPI_Wait(&request[7], &status); //blocks and waits for destination process to receive data
        Kokkos::parallel_for(Nx, KOKKOS_LAMBDA(int i) {
            localArray(i,0) = recvBuffer4(i);
        });
        Kokkos::fence();
    }
    MPI_Wait(&request[6], &status); //blocks and waits for destination process to receive data   
}
// ####################### Done #################################


// ==============================================================
// ======== Gather all subarray data to the global array ========
// ==============================================================
void gatherData( Kokkos::View<double**, Kokkos::LayoutRight> globalArray, 
                 Kokkos::View<double**, Kokkos::LayoutRight> localArray, 
                 const int &Nx,  const int &Ny, 
                 const int &Npx, const int &Npy, 
                 const int &numGhost, const MPI_Comm& comm ) {
    int myRank, ibegin, jbegin, myCoords[2];
    MPI_Comm_rank(comm, &myRank);
    MPI_Cart_coords(comm, myRank, 2, myCoords);

    Kokkos::View<double*> buffer("buffer", (Nx-numGhost)*(Ny-numGhost));
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
            MPI_Recv(myCoords, 2, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(buffer.data(), (Nx-numGhost)*(Ny-numGhost), MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ibegin = myCoords[0]*(Nx-numGhost);
            jbegin = myCoords[1]*(Ny-numGhost);
            Kokkos::parallel_for("copy local arrays back to global array",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx-numGhost, Ny-numGhost}),
                KOKKOS_LAMBDA(int i, int j) {
                    int index = i + j*(Ny-numGhost); // Index of flattened array
                    int ii = ibegin + i;
                    int jj = jbegin + j;
                    globalArray(ii,jj) = buffer(index);
            });
            
        }
    }
    else {
        Kokkos::parallel_for("copy local arrays back to global array",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx-numGhost, Ny-numGhost}),
            KOKKOS_LAMBDA(int i, int j) {
                int index = i + j*(Ny-numGhost); // Index of flattened array
                buffer(index) = localArray(i+1,j+1);
        });
        // Send local data to rank 0
        MPI_Send(myCoords, 2 , MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(buffer.data(), (Nx-numGhost)*(Ny-numGhost) , MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    MPI_Barrier(comm);

}
// ####################### Done #################################


#endif