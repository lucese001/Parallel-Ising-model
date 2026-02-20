#pragma once
#include <vector>
#include <cstddef>
#include <omp.h>
#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include "prng_engine.hpp"
#include "utility.hpp"

using std::vector;
using std::binomial_distribution;
using std::mt19937_64;
using namespace std;

// Variabili globali esterne
extern int N_dim;
extern int world_rank;
extern int world_size;

inline int computeEnSite(const vector<int8_t>& conf,
                         size_t idx,
                         const vector<uint32_t>& stride_halo,
                         int N_dim) {
    int sum = 0;
    for (int d = 0; d < N_dim; ++d) {
        sum += conf[idx + stride_halo[d]];
        sum += conf[idx - stride_halo[d]];
    }
    return -sum*conf[idx];
}

// computeEn: somma parziale dell'energia sui siti specificati
// Restituisce la somma GREZZA (ogni coppia contata 2 volte).
// Il chiamante divide per 2 dopo aver sommato tutti i contributi.
long long computeEn(const vector<int8_t>& conf,
                           const vector<size_t>& sites,
                           const vector<uint32_t>& stride_halo,
                           int N_dim) {
    long long en = 0;
#pragma omp parallel for reduction(+:en)
    for (size_t i = 0; i < sites.size(); ++i) {
        en += computeEnSite(conf, sites[i], stride_halo, N_dim);
    }
    return en;
}

long long computeEn_rank(const vector<int8_t>& conf,
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
            E_local += computeEnSite(conf, halo_idx, stride_halo, N_dim);
        }
    }
    return E_local / 2;
}



long long compute_Mag_rank(const vector<int8_t>& conf,
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
            Mag_local += conf[halo_idx];
        }
    }
    return Mag_local;
}


/*
// computeMagnetization_local: magnetizzazione parziale sui siti specificati
long long compute_Mag_rank(const vector<int8_t>& conf,
                                            const vector<size_t>& sites) {
    long long mag = 0;
#pragma omp parallel for reduction(+:mag)
    for (size_t i = 0; i < sites.size(); ++i) {
        mag += conf[sites[i]];
    }
    return mag;
}*/

void initialize_configuration(vector<int8_t>& conf_local,
                                     size_t N_local,
                                     int N_dim,
                                     const vector<size_t>& local_L,
                                     const vector<size_t>& local_L_halo,
                                     const vector<size_t>& global_offset,
                                     const vector<size_t>& arr,
                                     uint32_t rng_seed,
                                     bool cold_start = false) {
    // Inizializza

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
            int8_t spin;
            if (cold_start) {
                spin = 1;
            } else {
                uint32_t rand_val = philox_rand(global_index, 0, rng_seed);
                spin = (rand_val >> 16) & 1 ? 1 : -1;
            }

            for (int d = 0; d < N_dim; ++d)
                coord_halo[d] = coord_local[d] + 1;
            size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
            conf_local[idx_halo] = spin;
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