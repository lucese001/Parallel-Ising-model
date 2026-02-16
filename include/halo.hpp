#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include "utility.hpp"

using std::vector;

// Tipo per passare i buffer per riferimento
struct HaloBuffers {
    vector<vector<int8_t>> send_minus, send_plus; 
    vector<vector<int8_t>> recv_minus, recv_plus; 
    
    void resize(int N_dim) {
        send_minus.resize(N_dim);  
        send_plus.resize(N_dim);
        recv_minus.resize(N_dim);
        recv_plus.resize(N_dim);
    }
};

// Tipo per passare per riferimento le informazioni delle facce
struct FaceInfo {
    vector<size_t> dims;
    vector<size_t> map;
};

struct FaceCache {
    size_t face_size;
    //nota: ogni vettore è composto da due vettori,
    // uno per il caso pari e uno per il caso dispari.
    std::array<vector<size_t>, 2> face_to_full;
    std::array<vector<size_t>, 2> idx_minus;      // Indici del boundary negativo (per fare Send)
    std::array<vector<size_t>, 2> idx_plus;       // Indici del boundary positivo (per fare Send)
    std::array<vector<size_t>, 2> idx_halo_minus; // Indici della regione halo negativa (per fare Recv/Write)
    std::array<vector<size_t>, 2> idx_halo_plus;  // Indici della regione halo positiva (per fare Recv/Write)
};

// Costruisce le informazioni delle facce per lo scambio halo
inline vector<FaceInfo> build_faces(const vector<size_t>& local_L, 
                                    int N_dim) {

    vector<FaceInfo> faces(N_dim);
    for (int d = 0; d < N_dim; ++d) {
        for (int k = 0; k < N_dim; ++k) {
            if (k == d) continue;
            faces[d].dims.push_back(local_L[k]);
            faces[d].map.push_back(k);
        }
    }
    return faces;
}

inline vector<FaceCache> 
build_face_cache(const vector<FaceInfo>& faces,
                 const vector<size_t>& local_L,
                 const vector<size_t>& local_L_halo,
                 const vector<size_t>& global_offset,
                 int N_dim)
{
    vector<FaceCache> cache(N_dim);
    vector<size_t> coord_face;
    vector<size_t> coord_full(N_dim);

    for (int d = 0; d < N_dim; ++d) {

        const vector<size_t>& face_dims    = faces[d].dims;
        const vector<size_t>& face_to_full = faces[d].map;

        size_t face_size = 1;
        for (size_t i = 0; i < face_dims.size(); ++i) {
            face_size *= face_dims[i];
        }

        cache[d].face_size = face_size;

        for (int p = 0; p < 2; ++p) {
            cache[d].idx_minus[p].reserve(face_size/2+1);
            cache[d].idx_plus[p].reserve(face_size/2+1);
            cache[d].idx_halo_minus[p].reserve(face_size/2+1);
            cache[d].idx_halo_plus[p].reserve(face_size/2+1);
        }

        coord_face.resize(face_dims.size());

        for (size_t i = 0; i < face_size; ++i) {

            //Coordinate del sito (locali rispetto alla faccia)
            index_to_coord(i, face_dims.size(),
                           face_dims.data(), coord_face.data());

            // Copia le coordinate del sito (locali rispetto
            //che adesso include peró pure l'halo)
            //face_to_full[j] indica le coordinate intese come
            // come es: [x,y,z] [0,1,2], eccetera...
            size_t base=0;
            for (size_t j = 0; j < face_to_full.size(); ++j){
                coord_full[face_to_full[j]] = coord_face[j] + 1;
                base+= coord_face[j]+global_offset[face_to_full[j]];
            }

            int par_pos_face=(base+global_offset[d]+local_L[d]-1) %2;
            int par_pos_face_halo=(base+global_offset[d]+local_L[d]) %2;
            int par_neg_face=(base+global_offset[d]) %2;
            int par_neg_face_halo=(base+global_offset[d]+1) %2;

            // faccia meno
            coord_full[d] = 1;
            size_t idx_inner_minus =
                coord_to_index(N_dim, local_L_halo.data(), 
                                coord_full.data());

            cache[d].idx_minus[par_neg_face].push_back(idx_inner_minus);

            // halo meno
            coord_full[d] = 0;
            cache[d].idx_halo_minus[par_neg_face_halo].push_back(
                coord_to_index(N_dim, local_L_halo.data(), 
                                coord_full.data())
            );

            // faccia più
            coord_full[d] = local_L[d];
            size_t idx_inner_plus =coord_to_index(N_dim, 
                                                    local_L_halo.data(), 
                                                    coord_full.data());

            cache[d].idx_plus[par_pos_face].push_back(idx_inner_plus);

            // halo più
            coord_full[d] = local_L[d] + 1;
            cache[d].idx_halo_plus[par_pos_face_halo].push_back(
                coord_to_index(N_dim, local_L_halo.data(), 
                                coord_full.data())
            );
        }
    }

    return cache;
}

// Overload esteso di build_face_cache: fonde il calcolo della face cache con la
// classificazione dei siti boundary (Red/Black + indici globali per il RNG).
// Rispetto all'overload base aggiunge:
//   arr              → dimensioni globali del reticolo (per calcolare global_idx)
//   boundary_sites   → output: indici halo dei siti boundary per parità
//   boundary_indices → output: indici globali dei siti boundary per parità
//
// I siti agli angoli/spigoli (boundary in 2+ dimensioni) sono deduplicati
// tramite un vettore seen[] di dimensione N_alloc = prod(local_L_halo).
//
// Il loop interno sul numero di siti per faccia è parallelizzato con OpenMP
// tramite accumulatori thread-local; il merge e la dedup sono seriali.
inline vector<FaceCache>
build_face_cache(const vector<FaceInfo>& faces,
                 const vector<size_t>& local_L,
                 const vector<size_t>& local_L_halo,
                 const vector<size_t>& global_offset,
                 const vector<size_t>& arr,
                 int N_dim,
                 vector<uint32_t> boundary_sites[2],
                 vector<uint32_t> boundary_indices[2])
{
    vector<FaceCache> cache(N_dim);

    // Stride globali per calcolare global_idx dei siti boundary
    vector<size_t> stride_global(N_dim);
    stride_global[0] = 1;
    for (int d = 1; d < N_dim; ++d)
        stride_global[d] = stride_global[d-1] * arr[d-1];

    // seen[halo_idx] = true se il sito è già in boundary_sites.
    // Evita inserimenti doppi per siti agli angoli/spigoli che compaiono
    // come boundary in più dimensioni.
    size_t N_alloc = 1;
    for (int d = 0; d < N_dim; ++d) N_alloc *= local_L_halo[d];
    vector<bool> seen(N_alloc, false);

    const int nT = omp_get_max_threads();

    for (int d = 0; d < N_dim; ++d) {

        const vector<size_t>& face_dims    = faces[d].dims;
        const vector<size_t>& face_to_full = faces[d].map;

        size_t face_size = 1;
        for (size_t i = 0; i < face_dims.size(); ++i)
            face_size *= face_dims[i];

        cache[d].face_size = face_size;

        // Pre-riserva spazio nella face cache
        for (int p = 0; p < 2; ++p) {
            cache[d].idx_minus [p].reserve(face_size / 2 + 1);
            cache[d].idx_plus [p].reserve(face_size / 2 + 1);
            cache[d].idx_halo_minus[p].reserve(face_size / 2 + 1);
            cache[d].idx_halo_plus [p].reserve(face_size / 2 + 1);
        }

        // Accumulatori thread-local per face cache e candidati boundary.
        // Indicizzati come [p * nT + tid] per evitare false sharing.
        vector<vector<size_t>>  th_idx_minus  (2 * nT);
        vector<vector<size_t>>  th_idx_plus (2 * nT);
        vector<vector<size_t>>  th_idx_halo_minus (2 * nT);
        vector<vector<size_t>>  th_idx_halo_plus  (2 * nT);
        vector<vector<uint32_t>> th_bsites (2 * nT);
        vector<vector<uint32_t>> th_bidx (2 * nT);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            // Buffer per-thread per le coordinate
            vector<size_t> coord_face_t(face_dims.size());
            vector<size_t> coord_full_t(N_dim);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < face_size; ++i) {

                index_to_coord(i, (int)face_dims.size(),face_dims.data(), coord_face_t.data());

                // Calcola la parità base e l'indice globale base
                // come somma dei contributi delle dimensioni k != d
                size_t base = 0;
                size_t base_global = 0;
                for (size_t j = 0; j < face_to_full.size(); ++j) {
                    int k  = (int)face_to_full[j];
                    coord_full_t[k] = coord_face_t[j] + 1; // coord halo = local + 1
                    size_t gc = coord_face_t[j] + global_offset[k];
                    base += gc;
                    base_global += gc * stride_global[k];
                }

                int par_pos_face = (int)((base + global_offset[d] + local_L[d] - 1) % 2);
                int par_pos_face_halo = (int)((base + global_offset[d] + local_L[d] ) % 2);
                int par_neg_face = (int)((base + global_offset[d] ) % 2);
                int par_neg_face_halo = (int)((base + global_offset[d] + 1 ) % 2);

                // faccia meno: coord_halo[d] = 1 (local coord[d] = 0)
                coord_full_t[d] = 1;
                size_t idx_inner_minus =coord_to_index(N_dim, local_L_halo.data(), coord_full_t.data());
                th_idx_minus[par_neg_face * nT + tid].push_back(idx_inner_minus);

                // global_coord[d] = 0 + global_offset[d]
                th_bsites   [par_neg_face * nT + tid].push_back((uint32_t)idx_inner_minus);
                th_bidx [par_neg_face * nT + tid].push_back((uint32_t)(base_global + global_offset[d]
                    * stride_global[d]));

                // halo meno: coord_halo[d] = 0
                coord_full_t[d] = 0;
                th_idx_halo_minus[par_neg_face_halo * nT + tid].push_back(
                    coord_to_index(N_dim, local_L_halo.data(), coord_full_t.data()));

                // faccia più: coord_halo[d] = local_L[d] (local coord[d] = local_L[d]-1)
                coord_full_t[d] = local_L[d];
                size_t idx_inner_plus =coord_to_index(N_dim, local_L_halo.data(), coord_full_t.data());
                th_idx_plus[par_pos_face * nT + tid].push_back(idx_inner_plus);

                // global_coord[d] = local_L[d]-1 + global_offset[d]
                th_bsites [par_pos_face * nT + tid].push_back((uint32_t)idx_inner_plus);
                th_bidx [par_pos_face * nT + tid].push_back((uint32_t)(base_global + (local_L[d]
                    - 1 + global_offset[d]) * stride_global[d]));

                // halo più: coord_halo[d] = local_L[d]+1
                coord_full_t[d] = local_L[d] + 1;
                th_idx_halo_plus[par_pos_face_halo * nT + tid].push_back(
                    coord_to_index(N_dim, local_L_halo.data(), coord_full_t.data()));
            }
        } // end omp parallel

        // Merge face cache (seriale)
        for (int p = 0; p < 2; ++p)
            for (int t = 0; t < nT; ++t) {
                cache[d].idx_minus [p].insert(cache[d].idx_minus [p].end(),
                th_idx_minus [p * nT + t].begin(), th_idx_minus [p * nT + t].end());
                cache[d].idx_plus [p].insert(cache[d].idx_plus [p].end(),
                th_idx_plus  [p * nT + t].begin(), th_idx_plus [p * nT + t].end());
                cache[d].idx_halo_minus[p].insert(cache[d].idx_halo_minus[p].end(),
                th_idx_halo_minus[p * nT + t].begin(), th_idx_halo_minus[p * nT + t].end());
                cache[d].idx_halo_plus [p].insert(cache[d].idx_halo_plus [p].end(),
                th_idx_halo_plus [p * nT + t].begin(), th_idx_halo_plus [p * nT + t].end());
            }

        // --- Merge boundary con dedup tramite seen[] (seriale) ---
        // La dedup è necessaria per i siti agli angoli/spigoli che compaiono
        // nell'iterazione su più dimensioni d. All'interno della stessa d
        // non ci sono duplicati (ogni posizione di faccia mappa su un sito unico).
        for (int p = 0; p < 2; ++p)
            for (int t = 0; t < nT; ++t) {
                const auto& bs = th_bsites[p * nT + t];
                const auto& bi = th_bidx  [p * nT + t];
                for (size_t j = 0; j < bs.size(); ++j) {
                    uint32_t h = bs[j];
                    if (!seen[h]) {
                        seen[h] = true;
                        boundary_sites  [p].push_back(h);
                        boundary_indices[p].push_back(bi[j]);
                    }
                }
            }

    } // fine loop su d

    return cache;
}

// Inizia lo scambio halo non-blocking
inline void start_halo_exchange(
    vector<uint64_t>& conf_local,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    const vector<vector<int>>& neighbors,
    MPI_Comm cart_comm,
    int N_dim,
    HaloBuffers& buffers,
    const vector<FaceInfo>& faces,
    vector<MPI_Request>& requests,
    const vector<FaceCache>& cache,
    int parity,
    bool debug_print)
{
    requests.clear();

    buffers.send_minus.resize(N_dim);
    buffers.send_plus.resize(N_dim);
    buffers.recv_minus.resize(N_dim);
    buffers.recv_plus.resize(N_dim);

    int rank;
    if (debug_print) {
        MPI_Comm_rank(cart_comm, &rank);
    }

    for (int d = 0; d < N_dim; ++d) {

        // Calcola la dimensione della faccia (tiene conto della parità)
        const size_t send_minus_size = cache[d].idx_minus[parity].size();
        const size_t send_plus_size  = cache[d].idx_plus[parity].size();
        const size_t recv_minus_size = cache[d].idx_halo_minus[parity].size();
        const size_t recv_plus_size  = cache[d].idx_halo_plus[parity].size();

        // Alloca i buffer
        buffers.send_minus[d].resize(send_minus_size);
        buffers.send_plus[d].resize(send_plus_size);
        buffers.recv_minus[d].resize(recv_minus_size);
        buffers.recv_plus[d].resize(recv_plus_size);

        // Prepara i buffer con le configurazioni
        // (solo siti della parità richiesta)
        for (size_t i = 0; i < send_minus_size; ++i) {
            buffers.send_minus[d][i] = get_spin(conf_local.data(),
                                cache[d].idx_minus[parity][i]);
        }

        for (size_t i = 0; i < send_plus_size; ++i) {
            buffers.send_plus[d][i] = get_spin(conf_local.data(),
                    cache[d].idx_plus[parity][i]);
        }

        int tag_minus = 100 + d;
        int tag_plus  = 200 + d;

        MPI_Request req;

        // Ricevi da vicino "dietro"
        MPI_Irecv(buffers.recv_minus[d].data(),
                  recv_minus_size, MPI_INT8_T,
                  neighbors[d][0], tag_plus,
                  cart_comm, &req);
        requests.push_back(req);

        // Ricevi da vicino "davanti"
        MPI_Irecv(buffers.recv_plus[d].data(),
                  recv_plus_size, MPI_INT8_T,
                  neighbors[d][1], tag_minus,
                  cart_comm, &req);
        requests.push_back(req);

        // Invia a vicino "dietro"
        MPI_Isend(buffers.send_minus[d].data(),
                  send_minus_size, MPI_INT8_T,
                  neighbors[d][0], tag_minus,
                  cart_comm, &req);
        requests.push_back(req);

        // Invia a vicino "davanti"
        MPI_Isend(buffers.send_plus[d].data(),
                  send_plus_size, MPI_INT8_T,
                  neighbors[d][1], tag_plus,
                  cart_comm, &req);
        requests.push_back(req);
    }
}


// Scrive i dati ricevuti nelle regioni halo
void write_halo_data(
    vector<uint64_t>& conf_local,
    const HaloBuffers& buffers,
    const vector<FaceInfo>& faces,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    int N_dim,
    const vector<FaceCache>& cache,
    int parity,
    vector<MPI_Request>& requests)
{
    // Aspetta la finalizzazione della comunicazione MPI
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();
    int rank;

    for (int d = 0; d < N_dim; ++d) {

        // Determina la dimensione della faccia (considerando la parità)
        const size_t halo_minus_size = cache[d].idx_halo_minus[parity].size();
        const size_t halo_plus_size  = cache[d].idx_halo_plus[parity].size();

        for (size_t i = 0; i < halo_minus_size; ++i) {

            // Scrive i dati ricevuti negli halo meno
            set_spin(conf_local.data(), cache[d].idx_halo_minus[parity][i], buffers.recv_minus[d][i]);
        }
        for (size_t i = 0; i < halo_plus_size; ++i) {

            // Scrive i dati ricevuti negli halo più
            set_spin(conf_local.data(), cache[d].idx_halo_plus[parity][i], buffers.recv_plus[d][i]);
        }
    }
}

// Calcola gli indici dei vicini usando la topologia cartesiana MPI
inline void halo_index(MPI_Comm cart_comm, int N_dim,
                      vector<vector<int>>& neighbors) {
    neighbors.resize(N_dim);
    for (int d = 0; d < N_dim; ++d) {
        neighbors[d].resize(2);
        MPI_Cart_shift(cart_comm, d, 1, &neighbors[d][0], &neighbors[d][1]);
    }
}

