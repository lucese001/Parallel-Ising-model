#pragma once
#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <omp.h>

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
                           vector<size_t> bulk_indices[2],
                           vector<uint32_t> boundary_sites[2],
                           vector<size_t> boundary_indices[2])
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
            boundary_indices[parity].push_back(global_idx);
        } else {
            bulk_sites[parity].push_back((uint32_t)halo_idx);
            bulk_indices[parity].push_back(global_idx);
        }
    }
}

// classify_bulk: classifica in parallelo SOLO i siti bulk (non boundary) in Red/Black.
// Usata dal path IDX_ALLOC in sostituzione della parte bulk di classify_sites.
// Loop "per righe" lungo dim 0 (come computeEn_rank), limitato ai soli siti
// interni: coord locale di ogni dim in [1, local_L[d]-2].
inline void classify_bulk(
    int N_dim,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    const vector<size_t>& global_offset,
    const vector<size_t>& arr,
    vector<uint32_t> bulk_sites[2],
    vector<size_t>   bulk_indices[2])
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
    stride_halo_loc  [0] = 1;
    stride_global_loc[0] = 1;
    for (int d = 1; d < N_dim; ++d) {
        stride_halo_loc  [d] = stride_halo_loc  [d-1] * local_L_halo[d-1];
        stride_global_loc[d] = stride_global_loc[d-1] * arr          [d-1];
    }

    const int nT = omp_get_max_threads();

    // Accumulatori thread-local: indicizzati come [p * nT + tid]
    vector<vector<uint32_t>> th_bulk_sites(2 * nT);
    vector<vector<size_t>>   th_bulk_idx  (2 * nT);

    // Loop parallelo sulle "bulk rows":
    // ogni row corrisponde a una combinazione fissa delle coord nelle dim 1..N-1,
    // tutte nel range interno [1, local_L[d]-2].
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < n_bulk_rows; ++row) {
        int    tid         = omp_get_thread_num();
        size_t tmp         = row;
        size_t base_halo   = 0;
        size_t base_global = 0;
        size_t parity_rest = 0;  // sum(gc_d per d=1..N-1)

        // Decodifica le coordinate delle dim 1..N-1 nel range [1, local_L[d]-2]
        for (int d = 1; d < N_dim; ++d) {
            size_t xd    = (tmp % (local_L[d] - 2)) + 1; // coord locale in [1, L[d]-2]
            tmp         /= (local_L[d] - 2);
            size_t gc    = xd + global_offset[d];
            base_halo   += (xd + 1) * stride_halo_loc  [d]; // coord halo = xd+1
            base_global += gc       * stride_global_loc[d];
            parity_rest += gc;
        }

        // Loop sui siti bulk lungo dim 0: x0 in [1, local_L[0]-2]
        for (size_t x0 = 1; x0 <= local_L[0] - 2; ++x0) {
            // stride_halo_loc[0] = 1 → halo_idx = base_halo + (x0+1)
            size_t halo_idx   = base_halo + x0 + 1;
            size_t gc0        = x0 + global_offset[0];
            size_t global_idx = base_global + gc0 * stride_global_loc[0];
            int    parity     = (int)((parity_rest + gc0) % 2);

            th_bulk_sites[parity * nT + tid].push_back((uint32_t)halo_idx);
            th_bulk_idx  [parity * nT + tid].push_back(global_idx);
        }
    }

    // Merge seriale dei risultati thread-local
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
