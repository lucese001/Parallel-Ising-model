#pragma once
#include <vector>
#include <cstddef>
#include <omp.h>
#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <random>
#include "prng_engine.hpp"
#include "utility.hpp"

using std::vector;
using std::binomial_distribution;
using std::mt19937_64;
using namespace std;

// Variabili globali esterne (definite in new_ising.cpp)
extern int N_dim;
extern int world_rank;
extern int world_size;

inline int computeEnSite(const vector<int8_t>& conf,
                         size_t idx, //indice halo
                         const vector<size_t>& stride_halo,
                         int N_dim) {
    int en = 0;
    for (int d = 0; d < N_dim; ++d) {
        en -= conf[idx + stride_halo[d]] * conf[idx];
        en -= conf[idx - stride_halo[d]] * conf[idx];
    }
    return en;
}

/*inline int computeEnSiteDebug(const vector<int8_t>& conf, 
                         const size_t& iSite_local,
                         const vector<size_t>& local_L,
                         const vector<size_t>& local_L_halo,bool condPrint) {

  condPrint&=omp_get_thread_num()==0;
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
    if (condPrint){
      if(world_rank==0)
	{
	  master_cout<<"coord[x]"<<static_cast<size_t>(coord_site.data()[0])<<"\n";
	  master_cout<<"coord[y]"<<static_cast<size_t>(coord_site.data()[1])<<"\n";
	}
   }
    
    // Aggiungi offset +1 per l'halo (le celle interne iniziano da 1)
    for (int d = 0; d < N_dim; ++d) {
        coord_halo[d] = coord_site[d] + 1;
        if (condPrint){
            master_cout<<"coord_halo["<<d<<"]"<<coord_halo[d]<<"\n";
        }
    }
    if (condPrint){
        master_cout << "=== Configuration (with halo) ===" << "\n";
        master_cout << "   ";
        for (size_t x = 0; x < local_L_halo[0]; ++x) {
            master_printf("%zu ", x);
        }
        master_printf("\n");
        for (size_t y = 0; y < local_L_halo[1]; ++y) {
            master_printf("%zu: ", y);
            for (size_t x = 0; x < local_L_halo[0]; ++x) {
                size_t idx_halo = x + y * local_L_halo[0];
                // Mark the current site with brackets
                if (x == coord_halo[0] && y == coord_halo[1]) {
                    master_printf("[%c]", conf[idx_halo] > 0 ? '+' : '-');
                } else {
                    master_printf(" %c ", conf[idx_halo] > 0 ? '+' : '-');
                }
            }
            master_printf("\n");
        }
        master_cout << "Current site: (" << coord_halo[0] << ", " << coord_halo[1] << ")" << "\n";
    }



    // Indice nel conf_local (con halo)
    size_t idx_center = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
    
    int en = 0;
    for (int d = 0; d < N_dim; ++d) {
        // Vicino +1
        memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
        coord_neigh[d] = coord_halo[d] + 1;
        if (condPrint){
            master_cout<<"coord_neigh["<<d<<"]"<<coord_neigh[d]<<"\n";
        }
        size_t idx_plus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
        if (condPrint){
            master_cout<<"idx_plus:"<<  static_cast<size_t>(idx_plus) << "\n";
            master_cout<<"idx_center:"<<  static_cast<size_t>(idx_center) << "\n";
            master_cout<<"conf_idx_plus:"<<  static_cast<int>(conf[idx_plus]) << "\n";
            master_cout<<"conf_idx_center:"<<  static_cast<int>(conf[idx_center]) << "\n";
        }
        en -= conf[idx_plus] * conf[idx_center];
        if (condPrint){
            master_cout<<"en1 "<<en<<"\n";
        }
        
        // Vicino -1
        memcpy(coord_neigh.data(), coord_halo.data(), N_dim * sizeof(size_t));
        coord_neigh[d] = coord_halo[d] - 1;
        if (condPrint){
            master_cout<<"coord_neigh1["<<d<<"]"<<coord_neigh[d]<<"\n";
        }
        size_t idx_minus = coord_to_index(N_dim, local_L_halo.data(), coord_neigh.data());
        en -= conf[idx_minus] * conf[idx_center];
        if (condPrint){
            master_cout<<"idx_minus:"<<  static_cast<size_t>(idx_minus) << "\n";
            master_cout<<"idx_center:"<<  static_cast<size_t>(idx_center) << "\n";
            master_cout<<"conf_idx_minus:"<<  static_cast<int>(conf[idx_minus]) << "\n";
            master_cout<<"conf_idx_center:"<<  static_cast<int>(conf[idx_center]) << "\n";
            master_cout<<"en2 "<<en<<"\n";
        }  

    }
    
    return en;
}*/

// computeEn: somma parziale dell'energia sui siti specificati
// Restituisce la somma GREZZA (ogni coppia contata 2 volte).
// Il chiamante divide per 2 dopo aver sommato tutti i contributi.
inline long long computeEn(const vector<int8_t>& conf,
                           const vector<size_t>& sites,
                           const vector<size_t>& stride_halo,
                           int N_dim) {
    long long en = 0;
#pragma omp parallel for reduction(+:en)
    for (size_t i = 0; i < sites.size(); ++i) {
        en += computeEnSite(conf, sites[i], stride_halo, N_dim);
    }
    return en;
}

// computeMagnetization_local: magnetizzazione parziale sui siti specificati
inline long long computeMagnetization_local(const vector<int8_t>& conf,
                                            const vector<size_t>& sites) {
    long long mag = 0;
#pragma omp parallel for reduction(+:mag)
    for (size_t i = 0; i < sites.size(); ++i) {
        mag += conf[sites[i]];
    }
    return mag;
}

// Crea una configurazione iniziale casuale usando l'indice globale 
// (per garantire riproducibilità) indipendente dal numero di rank/thread
/*inline void initialize_configuration(vector<int8_t>& conf_local,
                                     size_t N_local,
                                     int N_dim,
                                     const vector<size_t>& local_L,
                                     const vector<size_t>& local_L_halo,
                                     const vector<size_t>& global_offset,
                                     const vector<size_t>& arr,
                                     PhiloxRNG& gen,
                                     uint64_t base_seed) {


//Debug
for (int rank = 0; rank < world_size; rank++) {
    if (world_rank == rank) {
        printf("[INIT START] Rank=%d, N_local=%zu\n", world_rank, N_local);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

    std::fill(conf_local.begin(), conf_local.end(), 0);
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
            uint32_t rand_val = gen.get1(global_index, 0, 0, false);
            int8_t spin = (rand_val & 1) ? 1 : -1;


            
            // Converti l'indice locale (senza halo) in indice con halo
            index_to_coord(i, N_dim, local_L.data(), coord_local.data());
            for (int d = 0; d < N_dim; ++d) {
                coord_halo[d] = coord_local[d] + 1;  // +1 per saltare l'halo
            }
            size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
            
            // Memorizza lo spin
            conf_local[idx_halo] = spin;
        }
    }
    // Debug ordinato per rank
    for (int r = 0; r < world_size; ++r) {
        if (world_rank == r) {
            printf("[INIT] Rank=%d, N_local=%zu, first 5 global_idx: ", world_rank, N_local);
            for (size_t i = 0; i < conf_local.size(); ++i) {
                printf("%zu(%c) ", de[i], debug_spins[i] > 0 ? '+' : '-');
            }
            printf("\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}*/


inline void initialize_configuration(vector<int8_t>& conf_local,
                                     size_t N_local,
                                     int N_dim,
                                     const vector<size_t>& local_L,
                                     const vector<size_t>& local_L_halo,
                                     const vector<size_t>& global_offset,
                                     const vector<size_t>& arr,
                                     PhiloxRNG& gen,
                                     uint64_t base_seed) {
    // Inizializza
    std::fill(conf_local.begin(), conf_local.end(), 0);
    
    // Buffer per debug (fuori dal parallelo)
    vector<size_t> debug_global_idx;
    vector<int8_t> debug_spins;
    
    #pragma omp parallel
    {
        vector<size_t> coord_local(N_dim);
        vector<size_t> coord_halo(N_dim);
        vector<size_t> coord_global(N_dim);
        
        #pragma omp for
        for (size_t i = 0; i < N_local; ++i) {

            size_t global_index = compute_global_index(i, local_L, global_offset, arr, N_dim,
                                                       coord_local.data(), coord_global.data());
            uint32_t rand_val = gen.get1(global_index, 0, 0, false);
            int8_t spin = (rand_val & 1) ? 1 : -1;
            
            // Salva per debug (solo primi 5)
            #pragma omp critical
            {
                if (debug_global_idx.size() < 5) {
                    debug_global_idx.push_back(global_index);
                    debug_spins.push_back(spin);
                }
            }
            
            // Converti in coordinate con halo
            index_to_coord(i, N_dim, local_L.data(), coord_local.data());
            for (int d = 0; d < N_dim; ++d) {
                coord_halo[d] = coord_local[d] + 1;
            }
            size_t idx_halo = coord_to_index(N_dim, local_L_halo.data(), coord_halo.data());
            conf_local[idx_halo] = spin;
        }
    }
    
    // STAMPA ORDINATA PER RANK (fuori dal parallelo)
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
