#pragma once
#include <vector>
#include <cstddef>
#include <cstring>
#include <random>
#include "prng_engine.hpp"
#include "utility.hpp"

using std::vector;
using std::binomial_distribution;
using std::mt19937_64;

// Variabili globali esterne (definite in new_ising.cpp)
extern size_t N_dim;

// computeEnSite: energia locale attorno a iSite
inline int computeEnSite(const vector<int>& conf, 
                         const size_t& iSite_local,
                         const vector<size_t>& local_L,
                         const vector<size_t>& local_L_halo) {
    
    static thread_local vector<size_t> coord_site(N_dim);
    static thread_local vector<size_t> coord_halo(N_dim);
    static thread_local vector<size_t> coord_neigh(N_dim);
    
    if (coord_site.size() != N_dim) {
        coord_site.resize(N_dim);
        coord_halo.resize(N_dim);
        coord_neigh.resize(N_dim);
    }
    
    // Converti iSite_local (senza halo) in coordinate locali
    index_to_coord(iSite_local, N_dim, local_L.data(), coord_site.data());
    
    // Aggiungi offset +1 per l'halo (le celle interne iniziano da 1)
    for (size_t d = 0; d < N_dim; ++d) {
        coord_halo[d] = coord_site[d] + 1;
    }
    
    // Indice nel conf_local (con halo)
    size_t idx_center = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
    
    int en = 0;
    for (size_t d = 0; d < N_dim; ++d) {
        // Vicino +1
        memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
        coord_neigh[d] = coord_halo[d] + 1;
        size_t idx_plus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
        en -= conf[idx_plus] * conf[idx_center];
        
        // Vicino -1
        memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
        coord_neigh[d] = coord_halo[d] - 1;
        size_t idx_minus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
        en -= conf[idx_minus] * conf[idx_center];
    }
    
    return en;
}

// computeEn: energia totale (riduzione parallela)
inline int computeEn(const vector<int>& conf, size_t N_local,
                     const vector<size_t>& local_L,
                     const vector<size_t>& local_L_halo) {
    long long en = 0;
#pragma omp parallel for reduction(+:en)
    for (size_t iSite = 0; iSite < N_local; ++iSite) {
        en += computeEnSite(conf, iSite, local_L, local_L_halo);
    }
    return (int)(en / 2);
}

// computeMagnetization_local: magnetizzazione locale
inline double computeMagnetization_local(const vector<int>& conf, size_t N_local,
                                         const vector<size_t>& local_L,
                                         const vector<size_t>& local_L_halo) {
    long long mag = 0;
    
#pragma omp parallel reduction(+:mag)
    {
        vector<size_t> coord_local(N_dim);
        vector<size_t> coord_halo(N_dim);
        
#pragma omp for
        for (size_t iSite = 0; iSite < N_local; ++iSite) {
            index_to_coord(iSite, N_dim, local_L.data(), coord_local.data());
            for (size_t d = 0; d < N_dim; ++d) {
                coord_halo[d] = coord_local[d] + 1;
            }
            size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
            mag += conf[idx_halo];
        }
    }
    return (double) mag;
}

// initialize_configuration: crea configurazione iniziale casuale (prng_engine)
inline void initialize_configuration(vector<int>& conf_local, 
                                     size_t N_alloc,
                                     prng_engine& gen) {
    for (size_t i = 0; i < N_alloc; ++i) {
        conf_local[i] = binomial_distribution<int>(1, 0.5)(gen) * 2 - 1;
    }
}

#ifndef PARALLEL_RNG
// Overload per vector<mt19937_64>
inline void initialize_configuration(vector<int>& conf_local, 
                                     size_t N_alloc,
                                     vector<mt19937_64>& gen) {
    for (size_t i = 0; i < N_alloc; ++i) {
        conf_local[i] = binomial_distribution<int>(1, 0.5)(gen[i]) * 2 - 1;
    }
}
#endif
