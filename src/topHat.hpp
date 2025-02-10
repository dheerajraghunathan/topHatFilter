#ifndef TOPHAT_HPP
#define TOPHAT_HPP

#include <vector>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

using namespace std;

// ==============================================================
// ====== Function to generate random values to a 2D array ======
// ==============================================================
void generateRandomData(Kokkos::View<double**, Kokkos::LayoutRight> array, const int &Nx, const int &Ny) {
    // Create random number pool
    Kokkos::Random_XorShift64_Pool<> rand_pool(/*seed=*/12345);

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
// ===== Function to apply top-hat filter with Periodic BC ======
// ==============================================================
void applyTopHatFilterPeriodicBC(Kokkos::View<double**, Kokkos::LayoutRight> array, const int &Nx, const int &Ny) {
    Kokkos::View<double**, Kokkos::LayoutRight> tempArray("tempArray", Nx, Ny);
    Kokkos::deep_copy(tempArray, array);          // Copying to a temp array
    const double inv9 = 1.0/9.0;
    Kokkos::parallel_for("TopHatFilter",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int i, int j) {
            int i_east  = i+1;
            int i_west  = i-1;
            int j_north = j+1;
            int j_south = j-1;

            if (i==0)    i_west  = Nx-1;
            if (i==Nx-1) i_east  = 0;
            if (j==0)    j_south = Ny-1;
            if (j==Ny-1) j_north = 0;
 
            double centre    = tempArray(i,j);
            double east      = tempArray(i_east,j);
            double west      = tempArray(i_west,j);
            double north     = tempArray(i,j_north);
            double south     = tempArray(i,j_south);
            double northeast = tempArray(i_east,j_north);
            double northwest = tempArray(i_west,j_north);
            double southeast = tempArray(i_east,j_south);
            double southwest = tempArray(i_west,j_south);

            array(i,j)  =  inv9 * ( centre + east + west + north + south +
                                     northeast + northwest + southeast + southwest ) ;
    });

}
// ####################### Done #################################

// ==============================================================
// ======= Function to apply top-hat filter with Zero BC  =======
// ==============================================================
void applyTopHatFilterZeroBC(Kokkos::View<double**, Kokkos::LayoutRight> array, const int &Nx, const int &Ny) {
    Kokkos::View<double**, Kokkos::LayoutRight> tempArray("tempArray", Nx+2, Ny+2);
    Kokkos::deep_copy(tempArray, 0.0); // Temp array initialised to zero
    Kokkos::parallel_for("TopHatFilter",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int i, int j) {
            tempArray(i+1,j+1) = array(i,j); // Offset by 1 to create zero borders
    });
    const double inv9 = 1.0/9.0;
    Kokkos::parallel_for("TopHatFilter",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int i, int j) {
            // Indices of tempArray
            int ii = i+1;  
            int jj = j+1;

            int i_east  = ii+1;
            int i_west  = ii-1;
            int j_north = jj+1;
            int j_south = jj-1;
            
            double centre    = tempArray(ii,jj);
            double east      = tempArray(i_east,jj);
            double west      = tempArray(i_west,jj);
            double north     = tempArray(ii,j_north);
            double south     = tempArray(ii,j_south);
            double northeast = tempArray(i_east,j_north);
            double northwest = tempArray(i_west,j_north);
            double southeast = tempArray(i_east,j_south);
            double southwest = tempArray(i_west,j_south);

            array(i,j)  =  inv9 * ( centre + east + west + north + south +
                                     northeast + northwest + southeast + southwest ) ;
    });

}
// ####################### Done #################################

#endif