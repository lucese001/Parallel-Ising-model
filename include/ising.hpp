#pragma once
#include <vector>
#include <cstddef>
#include <omp.h>
#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <random>
#include "utility.hpp"

using std::vector;
using namespace std;

// Variabili globali esterne (definite in main.cpp)
extern int N_dim;
extern int world_rank;
extern int world_size;

inline int computeEnSite(const uint64_t* conf,
                         size_t idx,
                         const vector<uint32_t>& stride_halo,
                         int N_dim) {

    int center_bit = (conf[idx >> 6] >> (idx & 63)) & 1;
    int n_up = 0;
    for (int d = 0; d < N_dim; ++d) {
        size_t ip = idx + stride_halo[d];
        size_t im = idx - stride_halo[d];
        n_up += (conf[ip >> 6] >> (ip & 63)) & 1;
        n_up += (conf[im >> 6] >> (im & 63)) & 1;
    }
    return -(2 * center_bit - 1) * (2 * n_up - 2 * N_dim);
}

long long computeEn_rank(const vector<uint64_t>& conf,
                         const vector<uint32_t>& stride_halo,
                         const vector<size_t>& local_L,
                         int N_dim) 
{
    long long E_local = 0;
    
    size_t n_rows_all = 1;
    for (int d = 1; d < N_dim; d++) 
        n_rows_all *= local_L[d];

    #pragma omp parallel for schedule(static) reduction(+:E_local)
    for (size_t row = 0; row < n_rows_all; row++) {
        size_t base_halo = 0;
        size_t tmp = row;
        for (int d = 1; d < N_dim; d++) {
            size_t x_d = tmp % local_L[d];
            tmp /= local_L[d];
            base_halo += (x_d + 1) * stride_halo[d];
        }
        for (size_t x0 = 0; x0 < local_L[0]; x0++) {
            size_t halo_idx = base_halo + (x0 + 1);
            E_local += computeEnSite(conf.data(), halo_idx, stride_halo, N_dim);
        }
    }
    return E_local / 2;
}



long long compute_Mag_rank(const vector<uint64_t>& conf,
                           const vector<uint32_t>& stride_halo,
                           const vector<size_t>& local_L,
                           int N_dim)
{
    long long Mag_local = 0;
    
    size_t n_rows_all = 1;
    for (int d = 1; d < N_dim; d++) 
        n_rows_all *= local_L[d];

    #pragma omp parallel for schedule(static) reduction(+:Mag_local)
    for (size_t row = 0; row < n_rows_all; row++) {
        size_t base_halo = 0;
        size_t tmp = row;
        for (int d = 1; d < N_dim; d++) {
            size_t x_d = tmp % local_L[d];
            tmp /= local_L[d];
            base_halo += (x_d + 1) * stride_halo[d];
        }
        for (size_t x0 = 0; x0 < local_L[0]; x0++) {
            size_t halo_idx = base_halo + (x0 + 1);
            Mag_local += get_spin(conf.data(), halo_idx);
        }
    }
    return Mag_local;
}


void initialize_configuration(vector<uint64_t>& conf_local,
                                     size_t N_local,
                                     int N_dim,
                                     const vector<size_t>& local_L,
                                     const vector<size_t>& local_L_halo,
                                     const vector<size_t>& global_offset,
                                     const vector<size_t>& arr,
                                     uint32_t rng_seed) {
    
    // Inizializza tutti i siti a 0
    #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < conf_local.size(); ++i) conf_local[i] = 0;
    
    #ifdef DEBUG
    vector<size_t> debug_global_idx;
    vector<int8_t> debug_spins;
    #endif

    #pragma omp parallel
    {
        vector<size_t> coord_local(N_dim);
        vector<size_t> coord_halo(N_dim);
        vector<size_t> coord_global(N_dim);

        #pragma omp for
        for (size_t i = 0; i < N_local; ++i) {

            size_t global_index = compute_global_index(i, local_L, global_offset, arr, N_dim,
                                                       coord_local.data(), coord_global.data());
            uint32_t rand_val = philox_rand(global_index, 0, rng_seed);
            int8_t spin = (rand_val & 1) ? 1 : -1;

            #ifdef DEBUG
            #pragma omp critical
            {
                if (debug_global_idx.size() < 5) {
                    debug_global_idx.push_back(global_index);
                    debug_spins.push_back(spin);
                }
            }
            #endif

            for (int d = 0; d < N_dim; ++d)
                coord_halo[d] = coord_local[d] + 1;
            size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), 
                                            coord_halo.data());
            // In parallelo ma con pragma omp atomic (per evitare race condition
            // in scrittura).
            set_spin(conf_local.data(), idx_halo, spin);
        }
    }
    
    // Stampa ordinata per rank
    #ifdef DEBUG
        for (int r = 0; r < world_size; ++r) {
            if (world_rank == r) {
                printf("[INIT] Rank=%d, N_local=%zu, first 5 global_idx: ", world_rank, N_local);
                for (size_t i = 0; i < debug_global_idx.size(); ++i) {
                    printf("%zu(%c) ", debug_global_idx[i], debug_spins[i] > 0 ? '+' : '-');
                }
                printf("\n");
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD); // Aspetta che questo rank finisca
        }
    #endif
}