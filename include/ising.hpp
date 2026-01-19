#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <cassert>
#include "prng_engine.hpp"
#include "utility.hpp"

using std::vector;
using std::binomial_distribution;
using std::mt19937_64;
using namespace std;

// Variabili globali esterne (definite in new_ising.cpp)
extern size_t N_dim;

// computeEnSite: energia locale attorno a iSite
// is_bulk: true per siti bulk (non accedono all'halo), false per siti boundary
inline int computeEnSite(const vector<int8_t>& conf,
                         const size_t& iSite_local,
                         const vector<size_t>& local_L,
                         const vector<size_t>& local_L_halo,
                         bool is_bulk) {

    static thread_local vector<size_t> coord_site(N_dim);
    static thread_local vector<size_t> coord_halo(N_dim);
    static thread_local vector<size_t> coord_neigh(N_dim);

    if (coord_site.size() != N_dim) {
        coord_site.resize(N_dim);
        coord_halo.resize(N_dim);
        coord_neigh.resize(N_dim);
    }
    if (is_bulk) {
        // BULK: Converti iSite_local (senza halo) in coordinate locali
        size_t idx_center = iSite_local;
        index_to_coord(iSite_local, N_dim, local_L.data(), coord_site.data());
        int en = 0;
        for (size_t d = 0; d < N_dim; ++d) {
            // Vicino +1
            memcpy(coord_neigh.data(), coord_site.data(), N_dim * sizeof(size_t));
            coord_neigh[d] = coord_site[d] + 1;
            size_t idx_plus = coord_to_index(N_dim, local_L.data(), coord_neigh.data());
            en -= conf[idx_plus] * conf[idx_center];

            // Vicino -1
            memcpy(coord_neigh.data(), coord_site.data(), N_dim * sizeof(size_t));
            coord_neigh[d] = coord_site[d] - 1;
            size_t idx_minus = coord_to_index(N_dim, local_L.data(), coord_neigh.data());
            en -= conf[idx_minus] * conf[idx_center];
        }
        return en;
    } else {
        // BOUNDARY: Aggiungi offset +1 per l'halo (le celle interne iniziano da 1)
        index_to_coord(iSite_local, N_dim, local_L.data(), coord_site.data());
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
}
inline int computeEnSiteDebug(const vector<int8_t>& conf,
                         const size_t& iSite_local,
                         const vector<size_t>& local_L,
                         const vector<size_t>& local_L_halo,
                         bool condPrint,
                         bool is_bulk) {

    static thread_local vector<size_t> coord_site(N_dim);
    static thread_local vector<size_t> coord_halo(N_dim);
    static thread_local vector<size_t> coord_neigh(N_dim);

    if (coord_site.size() != N_dim) {
        coord_site.resize(N_dim);
        coord_halo.resize(N_dim);
        coord_neigh.resize(N_dim);
    }





    if (is_bulk) {
        size_t idx_center = iSite_local;
        index_to_coord(iSite_local, N_dim, local_L.data(), coord_site.data());
            if (condPrint){
            cout<<"coord[x]"<<static_cast<size_t>(coord_site[0])<<endl;
            cout<<"coord[y]"<<static_cast<size_t>(coord_site[1])<<endl;
            }
        int en = 0;
        for (size_t d = 0; d < N_dim; ++d) {

            // Vicino +1
            memcpy(coord_neigh.data(), coord_site.data(), N_dim * sizeof(size_t));
            coord_neigh[d] = coord_site[d] + 1;
            if (condPrint){
                for (size_t i= 0 ;i<N_dim;i++){
                    cout<<"coord_neigh["<<i<<"]"<<coord_neigh[i]<<endl;
                }
            }
            size_t idx_plus = coord_to_index(N_dim, local_L.data(), coord_neigh.data());
            en -= conf[idx_plus] * conf[idx_center];
                    if (condPrint){
            cout<<"idx_plus:"<<  static_cast<size_t>(idx_plus) << endl;
            cout<<"idx_center:"<<  static_cast<size_t>(idx_center) << endl;
            cout<<"conf_idx_plus:"<<  static_cast<int>(conf[idx_plus]) << endl;
            cout<<"conf_idx_center:"<<  static_cast<int>(conf[idx_center]) << endl;
            cout<<"en(+1): "<< en<<endl;
        }

            // Vicino -1
            memcpy(coord_neigh.data(), coord_site.data(), N_dim * sizeof(size_t));
            coord_neigh[d] = coord_site[d] - 1;
            size_t idx_minus = coord_to_index(N_dim, local_L.data(), coord_neigh.data());
            en -= conf[idx_minus] * conf[idx_center];
            if (condPrint){
            cout<<"idx_minus:"<<  static_cast<size_t>(idx_minus) << endl;
            cout<<"idx_center:"<<  static_cast<size_t>(idx_center) << endl;
            cout<<"conf_idx_minus:"<<  static_cast<int>(conf[idx_minus]) << endl;
            cout<<"conf_idx_center:"<<  static_cast<int>(conf[idx_center]) << endl;
            cout<<"en(-1): "<< en<<endl;
        }
        }
        return en;
    } else {
        // BOUNDARY: Aggiungi offset +1 per l'halo (le celle interne iniziano da 1)
        index_to_coord(iSite_local, N_dim, local_L.data(), coord_site.data());
        for (size_t d = 0; d < N_dim; ++d) {
            coord_halo[d] = coord_site[d] + 1;
        }

        // Indice nel conf_local (con halo)
        size_t idx_center = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
            if (condPrint){
            cout<<"coord[x]"<<static_cast<size_t>(coord_site[0])<<endl;
            cout<<"coord[y]"<<static_cast<size_t>(coord_site[1])<<endl;
   }

        int en = 0;
        for (size_t d = 0; d < N_dim; ++d) {
            // Vicino +1
            memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
            coord_neigh[d] = coord_halo[d] + 1;
            if (condPrint){
                for (size_t i=0; i<N_dim;i++ ){
                    cout<<"coord_neigh["<<i<<"]"<<coord_neigh[i]<<endl;
                }
            }

            size_t idx_plus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
            en -= conf[idx_plus] * conf[idx_center];
            if (condPrint){
            cout<<"idx_plus:"<<  static_cast<size_t>(idx_plus) << endl;
            cout<<"idx_center:"<<  static_cast<size_t>(idx_center) << endl;
            cout<<"conf_idx_plus:"<<  static_cast<int>(conf[idx_plus]) << endl;
            cout<<"conf_idx_center:"<<  static_cast<int>(conf[idx_center]) << endl;
            cout<<"en(+1):"<<en<< endl;
        }

            // Vicino -1
            memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
            coord_neigh[d] = coord_halo[d] - 1;

            size_t idx_minus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
            en -= conf[idx_minus] * conf[idx_center];
            if (condPrint){
            cout<<"idx_minus:"<<  static_cast<size_t>(idx_minus) << endl;
            cout<<"idx_center:"<<  static_cast<size_t>(idx_center) << endl;
            cout<<"conf_idx_minus:"<<  static_cast<int>(conf[idx_minus]) << endl;
            cout<<"conf_idx_center:"<<  static_cast<int>(conf[idx_center]) << endl;
            cout<<"en(-1):"<<en<< endl;
        }
        }

        return en;
    }
}

// computeEn: energia totale (riduzione parallela)
inline int computeEn(const vector<int8_t>& conf, size_t N_local,
                     const vector<size_t>& local_L,
                     const vector<size_t>& local_L_halo) {
    long long en = 0;
#pragma omp parallel for reduction(+:en)
    for (size_t iSite = 0; iSite < N_local; ++iSite) {
        en += computeEnSite(conf, iSite, local_L, local_L_halo,false);
    }
    return (int)(en / 2);
}

// computeMagnetization_local: magnetizzazione locale
inline double computeMagnetization_local(const vector<int8_t>& conf, size_t N_local,
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

// Crea una configurazione iniziale casuale usando l'indice globale 
// (per garantire riproducibilità) indipendente dal numero di rank/thread
inline void initialize_configuration(vector<int8_t>& conf_local,
                                     size_t N_local,
                                     size_t N_dim,
                                     const vector<size_t>& local_L,
                                     const vector<size_t>& local_L_halo,
                                     const vector<size_t>& global_offset,
                                     const vector<size_t>& arr,
                                     uint64_t base_seed) {
    // Inizializza tutto a 0
    std::fill(conf_local.begin(), conf_local.end(), 0);
    
    #pragma omp parallel
    {
        // Ogni thread ha i suoi buffer per le coordinate
        vector<size_t> coord_local(N_dim);
        vector<size_t> coord_halo(N_dim);
        vector<size_t> coord_global(N_dim);  // buffer per compute_global_index
        
        #pragma omp for
        for (size_t i = 0; i < N_local; ++i) {

            size_t global_index = compute_global_index(i, local_L, global_offset, arr, N_dim,
                                                       coord_local.data(), coord_global.data());
            uint64_t site_seed = base_seed + global_index;
            prng_engine site_gen(site_seed);
            int8_t spin = (site_gen() & 1) ? 1 : -1;
            
            // Converti l'indice locale (senza halo) in indice con halo
            index_to_coord(i, N_dim, local_L.data(), coord_local.data());
            for (size_t d = 0; d < N_dim; ++d) {
                coord_halo[d] = coord_local[d] + 1;  // +1 per saltare l'halo
            }
            size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
            
            // Memorizza lo spin
            conf_local[idx_halo] = spin;
        }
    }
}
