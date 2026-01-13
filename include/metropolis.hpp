#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>
#include "prng_engine.hpp"
#include "utility.hpp"
#include "ising.hpp"

using namespace std;
using std::vector;
using std::binomial_distribution;
using std::mt19937_64;

// Variabili globali esterne (definite in main.cpp)
extern size_t N_dim;
extern size_t N;
extern double Beta;

// metropolis_update: esegue uno sweep Metropolis completo
// con aggiornamento a scacchiera sui siti specificati
#ifdef PARALLEL_RNG
inline void metropolis_update(vector<int8_t>& conf_local,
                              const vector<size_t>& sites, 
                              const vector<size_t>& sites_global_indices,
                              const vector<size_t>& local_L,
                              const vector<size_t>& local_L_halo,
                              prng_engine& gen, int iConf,
                              size_t nThreads, size_t N_local) {
#else
inline void metropolis_update(vector<int8_t>& conf_local,
                              const vector<size_t>& sites, 
                              const vector<size_t>& sites_global_indices,
                              const vector<size_t>& local_L,
                              const vector<size_t>& local_L_halo,
                              vector<mt19937_64>& gen, int iConf,
                              size_t nThreads, size_t N_local) {
#endif
    
    vector<size_t> coord_buf(N_dim);
    vector<size_t> coord_tmp(N_dim);
    
    for (int par = 0; par < 2; ++par) { //Aggiornamento a scacchiera
#ifdef PARALLEL_RNG
#pragma omp parallel firstprivate(coord_buf, coord_tmp)
        {
            const size_t iThread = omp_get_thread_num();
            const size_t chunkSize = (sites.size() + nThreads - 1) / nThreads;
            const size_t beg = chunkSize * iThread;
            const size_t end = std::min(sites.size(), beg + chunkSize);            
            for (size_t idx = beg; idx < end; ++idx) {
                size_t iSite = sites[idx];
                size_t global_idx = sites_global_indices[idx]; 
#else
#pragma omp parallel for firstprivate(coord_buf, coord_tmp)
            for (size_t idx = 0; idx < sites.size(); ++idx) {
                size_t iSite = sites[idx];
                size_t global_idx = sites_global_indices[idx]; 
                mt19937_64& genView = gen[iSite];
#endif
                index_to_coord(iSite, N_dim, local_L.data(), 
                               coord_buf.data());
                size_t sum = 0;
                for (size_t d = 0; d < N_dim; ++d) sum += coord_buf[d];
                size_t pSite = sum % 2;

                if (par == (int)pSite) {
#ifdef PARALLEL_RNG
                    // discard basato sull'indice globale
                    prng_engine genView = gen;
                    genView.discard(2 * 2 * (global_idx / 2 + N / 2 * (par + 2 * iConf)));
#endif
                    
                    for (size_t d = 0; d < N_dim; ++d) {
                        coord_tmp[d] = coord_buf[d] + 1;
                    }
                    size_t iSite_halo = coord_to_index(N_dim, local_L_halo.data(), 
                                                       coord_tmp.data());
                    
                    const int8_t oldVal = conf_local[iSite_halo];
                    const int enBefore = computeEnSite(conf_local, iSite, 
                                                       local_L, local_L_halo);

                    conf_local[iSite_halo] = (int8_t)(binomial_distribution<int>(1, 0.5)(genView) * 2 - 1);

                    const int enAfter = computeEnSite(conf_local, iSite, 
                                                      local_L, local_L_halo);
                    const int eDiff = enAfter - enBefore;
                    const double pAcc = std::min(1.0, exp(-Beta * (double)eDiff));
                    const int acc = binomial_distribution<int>(1, pAcc)(genView);

                    if (!acc) conf_local[iSite_halo] = oldVal;
                }
#ifdef PARALLEL_RNG
            }
        }
#else
            }
#endif
    }
}
