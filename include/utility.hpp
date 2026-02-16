#pragma once
#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <omp.h>
#include <cstdint>
#ifndef R123_ASSERT
#include <cassert>
#define R123_ASSERT(x) assert(x)
#endif

#include "include/Random123/philox.h"

using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;


extern int world_rank;

template <typename...Args>
auto master_printf(const char* fmt,
		   const Args&...args)
{
  if(world_rank==0)
    printf(fmt,args...);
}

struct Couter
{
  template <typename T>
  Couter& operator<<(T&& t)
  {
    if(world_rank==0)
      std::cout<<t;

    return *this;
  }
};

inline Couter master_cout;

// Creazione di un timer per misurare le prestazioni
struct timer {
    static timer timerCost;

    static auto now() { return high_resolution_clock::now(); }

    size_t tot = 0;
    size_t n = 0;
    high_resolution_clock::time_point from;

    void start() { n++; from = now(); }
    void stop() { tot += duration_cast<nanoseconds>(now()-from).count(); }

    double get(bool sub = true) {
        double res = (double) tot;
        if (sub && timerCost.n > 0)
            res -= timerCost.tot * (double)n / (double)timerCost.n;
        return res / 1e9;
    }
};

// Converte un indice lineare in coordinate multi-dimensionali
inline void index_to_coord(size_t index, int N_dim, const size_t *arr_ptr, size_t *coord_buf) {
    for (int d = 0; d < N_dim; ++d) {
        coord_buf[d] = index % arr_ptr[d];
        index /= arr_ptr[d];
    }
}

// Converte coordinate multi-dimensionali in un indice lineare
inline size_t coord_to_index(int N_dim, const size_t *arr_ptr, const size_t *coord_buf) {
    size_t index = 0;
    size_t mult = 1;
    for (int d = 0; d < N_dim; ++d) {
        index += coord_buf[d] * mult;
        mult *= arr_ptr[d];
    }
    return index;
}

// Calcola l'indice globale di un sito dato il suo indice locale
// NOTA: usa buffer pre-allocati per evitare allocazioni non-deterministiche in OpenMP
inline size_t compute_global_index(size_t iSite_local,
                                   const vector<size_t>& local_L,
                                   const vector<size_t>& global_offset,
                                   const vector<size_t>& arr,
                                   int N_dim,
                                   size_t* coord_local,
                                   size_t* coord_global) {
    // Converte l'indice locale in coordinate locali
    index_to_coord(iSite_local, N_dim, local_L.data(), coord_local);

    // Converte coordinate locali in coordinate globali
    for (int d = 0; d < N_dim; ++d) {
        coord_global[d] = coord_local[d] + global_offset[d];
    }

    // Converte coordinate globali in indice globale
    return coord_to_index(N_dim, arr.data(), coord_global);
}

// Overload semplificato per calcolare la parità dalle coordinate
// Ritorna la somma delle coordinate (usata per calcolare parità)
inline size_t compute_global_index(const vector<size_t>& coord_full) {
    size_t sum = 0;
    for (size_t d = 0; d < coord_full.size(); ++d) {
        sum += coord_full[d];
    }
    return sum;
}

// classify_sites: classifica TUTTI i siti in bulk/boundary e Red/Black
// Usata dal path IDX_ALLOC (indici bulk pre-allocati)

inline void classify_sites(size_t N_local, int N_dim,
                           const vector<size_t>& local_L,
                           const vector<size_t>& local_L_halo,
                           const vector<size_t>& global_offset,
                           const vector<size_t>& arr,
                           vector<uint32_t> bulk_sites[2],
                           vector<uint32_t> bulk_indices[2],
                           vector<uint32_t> boundary_sites[2],
                           vector<uint32_t> boundary_indices[2])
{
    // Precompute strides
    vector<size_t> stride_halo_sz(N_dim), stride_global(N_dim);
    stride_halo_sz[0] = 1;
    stride_global[0] = 1;
    for (int d = 1; d < N_dim; d++) {
        stride_halo_sz[d] = stride_halo_sz[d-1] * local_L_halo[d-1];
        stride_global[d] = stride_global[d-1] * arr[d-1];
    }

    vector<size_t> coord_buf(N_dim);

    for (size_t iSite = 0; iSite < N_local; ++iSite) {
        index_to_coord(iSite, N_dim, local_L.data(), coord_buf.data());

        bool is_boundary = false;
        size_t parity_sum = 0;
        size_t halo_idx = 0;
        size_t global_idx = 0;

        for (int d = 0; d < N_dim; ++d) {
            if (coord_buf[d] == 0 || coord_buf[d] == local_L[d] - 1)
                is_boundary = true;

            size_t gc = coord_buf[d] + global_offset[d];
            global_idx += gc * stride_global[d];
            parity_sum += gc;
            halo_idx += (coord_buf[d] + 1) * stride_halo_sz[d];
        }

        int parity = parity_sum % 2;

        if (is_boundary) {
            boundary_sites[parity].push_back((uint32_t)halo_idx);
            boundary_indices[parity].push_back((uint32_t)global_idx);
        } else {
            bulk_sites[parity].push_back((uint32_t)halo_idx);
            bulk_indices[parity].push_back((uint32_t)global_idx);
        }
    }
}

// classify_bulk: classifica in parallelo i siti bulk (non boundary) in Red/Black.
// Loop per righe lungo la dimensione 0, limitato ai soli siti interni: la loro coordinata
// locale in ogni dimensione è compresa tra [1, local_L[d]-2].

inline void classify_bulk(
    int N_dim,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    const vector<size_t>& global_offset,
    const vector<size_t>& arr,
    vector<uint32_t> bulk_sites[2],
    vector<uint32_t> bulk_indices[2])
{
    // Controlla se esistono siti bulk: ogni dimensione deve avere almeno 3 siti
    bool has_bulk = (local_L[0] >= 3);
    size_t n_bulk_rows = 1;

    for (int d = 1; d < N_dim; ++d) {
        if (local_L[d] < 3) { has_bulk = false; break; }
        n_bulk_rows *= local_L[d] - 2;  // combinazioni interne nelle dim 1..N-1
    }
    if (!has_bulk) return;

    // Strides nel layout halo e nel layout globale
    vector<size_t> stride_halo_loc  (N_dim);
    vector<size_t> stride_global_loc(N_dim);
    stride_halo_loc [0] = 1;
    stride_global_loc [0] = 1;
    for (int d = 1; d < N_dim; ++d) {
        stride_halo_loc  [d] = stride_halo_loc  [d-1] * local_L_halo[d-1];
        stride_global_loc[d] = stride_global_loc[d-1] * arr [d-1];
    }

    const int nT = omp_get_max_threads();

    // Accumulatori thread-local: indicizzati come [p * nT + tid]
    vector<vector<uint32_t>> th_bulk_sites(2 * nT);
    vector<vector<uint32_t>> th_bulk_idx  (2 * nT);

    // Loop parallelo sulle righe bulk: ogni riga corrisponde a una combinazione fissa delle 
    // coordinate nelle dim 1..N-1, nel range interno [1, local_L[d]-2]. Si itera su tutte le righe 
    // nelle direzioni ortogonali alla dimensione 0.

    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < n_bulk_rows; ++row) {
        int tid = omp_get_thread_num();
        size_t tmp = row;
        size_t base_halo = 0;
        size_t base_global = 0;
        size_t parity_rest = 0;  // sum(gc_d per d=1..N-1)

        // Decodifica le coordinate delle dim 1..N-1 nel range [1, local_L[d]-2]
        for (int d = 1; d < N_dim; ++d) {
            size_t xd  = (tmp % (local_L[d] - 2)) + 1; // coord locale in [1, L[d]-2]
            tmp /= (local_L[d] - 2);
            size_t gc = xd + global_offset[d];
            base_halo += (xd + 1) * stride_halo_loc  [d]; // coord halo = xd+1
            base_global += gc * stride_global_loc[d];
            parity_rest += gc;
        }

        // Loop sui siti bulk lungo dim 0: x0 in [1, local_L[0]-2]
        for (size_t x0 = 1; x0 <= local_L[0] - 2; ++x0) {
            size_t halo_idx = base_halo + x0 + 1;
            size_t gc0 = x0 + global_offset[0];
            size_t global_idx = base_global + gc0 * stride_global_loc[0];
            int  parity = (int)((parity_rest + gc0) % 2);

            th_bulk_sites[parity * nT + tid].push_back((uint32_t)halo_idx);
            th_bulk_idx  [parity * nT + tid].push_back((uint32_t)global_idx);
        }
    }

    // Merge seriale dei risultati locali ai thread
    for (int p = 0; p < 2; ++p)
        for (int t = 0; t < nT; ++t) {
            bulk_sites  [p].insert(bulk_sites  [p].end(),
                                   th_bulk_sites[p * nT + t].begin(),
                                   th_bulk_sites[p * nT + t].end());
            bulk_indices[p].insert(bulk_indices[p].end(),
                                   th_bulk_idx  [p * nT + t].begin(),
                                   th_bulk_idx  [p * nT + t].end());
        }
}


// Generatore di numeri casuali philox
inline uint32_t philox_rand(uint64_t global_idx, uint32_t iConf, uint32_t seed) {
    philox4x32_ctr_t ctr = {{
        (uint32_t)(global_idx),
        (uint32_t)(global_idx >> 32),
        (uint32_t)iConf,
        0
    }};
    philox4x32_key_t key = {{seed, 0}};
    return philox4x32(ctr, key).v[0];
}

// Leggi spin al sito i: restituisce +1 o -1
inline int8_t get_spin(const uint64_t* data, size_t i) {
    return ((data[i >> 6] >> (i & 63)) & 1) ? 1 : -1;
}
// Flippa spin al sito i (XOR: inverte il bit). Usato per il bulk
// (la maggior parte dei siti). Dato che atomic rallenta, assegniamo
// a ogni thread una riga, dove ogni riga ha un numero intero di parole
// Non tutti i bit delle parole avranno siti "veri", si usa un padding
// per avere un numero intero di word.

inline void flip_spin(uint64_t* data, size_t i) {
    const uint64_t mask = 1ULL << (i & 63);
    data[i >> 6] ^= mask;
}

// Flippa spin al sito i (XOR: inverte il bit)
// Atomic per evitare data race quando più thread modificano bit diversi
// nella stessa parola uint64_t. Atomic rallenta leggermente l'esecuzione,
// quindi viene usato solamente per i siti boundary (2 siti boundary
// possono trovarsi nella stessa word). Irrilevante perché i siti boundary
// sono pochi per reticoli grandi.

inline void flip_spin_atomic(uint64_t* data, size_t i) {
    const uint64_t mask = 1ULL << (i & 63);
    #pragma omp atomic
    data[i >> 6] ^= mask;
}

// Scrivi spin al sito i (+1 → bit=1, -1 → bit=0)
// Atomic per lo stesso motivo di flip_spin
inline void set_spin(uint64_t* data, size_t i, int8_t val) {
    const size_t word = i >> 6;
    const uint64_t mask = 1ULL << (i & 63);
    if (val > 0) {
        #pragma omp atomic
        data[word] |= mask;
    } else {
        const uint64_t nmask = ~mask;
        #pragma omp atomic
        data[word] &= nmask;
    }
}
