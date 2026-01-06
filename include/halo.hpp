#pragma once
#include <vector>
#include <cstddef>
#include <mpi.h>
#include "utility.hpp"

using std::vector;

// Tipo per passare i buffer per riferimento
struct HaloBuffers {
    vector<vector<int>> send_minus, send_plus; 
    vector<vector<int>> recv_minus, recv_plus; 
    
    void resize(size_t N_dim) {
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

// Costruisce le informazioni delle facce per lo scambio halo
inline vector<FaceInfo> build_faces(const vector<size_t>& local_L, size_t N_dim) {
    vector<FaceInfo> faces(N_dim);

    for (size_t d = 0; d < N_dim; ++d) {
        for (size_t k = 0; k < N_dim; ++k) {
            if (k == d) continue;
            faces[d].dims.push_back(local_L[k]);
            faces[d].map.push_back(k);
        }
    }
    return faces;
}

// Inizia lo scambio halo non-blocking
inline void start_halo_exchange(vector<int>& conf_local, 
                                const vector<size_t>& local_L,
                                const vector<size_t>& local_L_halo,
                                const vector<vector<int>>& neighbors, 
                                MPI_Comm cart_comm, size_t N_dim,
                                HaloBuffers& buffers,
                                const vector<FaceInfo>& faces,
                                vector<MPI_Request>& requests) {
    
    requests.clear();
    buffers.send_minus.resize(N_dim);
    buffers.send_plus.resize(N_dim);
    buffers.recv_minus.resize(N_dim);
    buffers.recv_plus.resize(N_dim);
    
    for (size_t d = 0; d < N_dim; ++d) {
        const vector<size_t>& face_dims = faces[d].dims;
        const vector<size_t>& face_to_full = faces[d].map;
        
        // Calcola face_size
        size_t face_size = 1;
        for (size_t x : face_dims) face_size *= x;
        
        // Resize dei buffer per questa dimensione
        buffers.send_minus[d].resize(face_size);
        buffers.send_plus[d].resize(face_size);
        buffers.recv_minus[d].resize(face_size);
        buffers.recv_plus[d].resize(face_size);
    
        vector<size_t> coord_face(face_dims.size());
        vector<size_t> coord_full(N_dim);
        
        // Prepara dati da inviare
        for (size_t i = 0; i < face_size; ++i) {
            index_to_coord(i, face_dims.size(), face_dims.data(), 
                          coord_face.data());
            
            for (size_t j = 0; j < face_to_full.size(); ++j) {
                coord_full[face_to_full[j]] = coord_face[j] + 1;
            }
            
            // Faccia negativa (bordo inferiore in dimensione d)
            coord_full[d] = 1;
            size_t idx_minus = coord_to_index(N_dim, local_L_halo.data(), 
                                             coord_full.data());
            buffers.send_minus[d][i] = conf_local[idx_minus];
            
            // Faccia positiva (bordo superiore in dimensione d)
            coord_full[d] = local_L[d];
            size_t idx_plus = coord_to_index(N_dim, local_L_halo.data(), 
                                            coord_full.data());
            buffers.send_plus[d][i] = conf_local[idx_plus];
        }
        
        int tag_minus = 100 + d;
        int tag_plus  = 200 + d;
        
        // Inizia comunicazioni non-blocking
        MPI_Request req;
        
        // Ricevi da vicino "dietro"
        MPI_Irecv(buffers.recv_minus[d].data(), face_size, MPI_INT,
                 neighbors[d][0], tag_plus, cart_comm, &req);
        requests.push_back(req);
        
        // Ricevi da vicino "davanti"
        MPI_Irecv(buffers.recv_plus[d].data(), face_size, MPI_INT,
                 neighbors[d][1], tag_minus, cart_comm, &req);
        requests.push_back(req);
        
        // Invia a vicino "dietro"
        MPI_Isend(buffers.send_minus[d].data(), face_size, MPI_INT,
                 neighbors[d][0], tag_minus, cart_comm, &req);
        requests.push_back(req);
        
        // Invia a vicino "davanti"
        MPI_Isend(buffers.send_plus[d].data(), face_size, MPI_INT,
                 neighbors[d][1], tag_plus, cart_comm, &req);
        requests.push_back(req);
    }
}

// Aspetta il completamento dello scambio halo
inline void finish_halo_exchange(std::vector<MPI_Request>& reqs) {
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();
}

// Scrive i dati ricevuti nelle regioni halo
inline void write_halo_data(vector<int>& conf_local,
                            const HaloBuffers& buffers,
                            const vector<FaceInfo>& faces,
                            const vector<size_t>& local_L,
                            const vector<size_t>& local_L_halo,
                            size_t N_dim) {
    
    for (size_t d = 0; d < N_dim; ++d) {
        const vector<size_t>& face_dims = faces[d].dims;
        const vector<size_t>& face_to_full = faces[d].map;
        
        size_t face_size = 1;
        for (size_t x : face_dims) face_size *= x;
        
        vector<size_t> coord_face(face_dims.size());
        vector<size_t> coord_full(N_dim);
        
        // Scrivi negli halo
        for (size_t i = 0; i < face_size; ++i) {
            index_to_coord(i, face_dims.size(), face_dims.data(), 
                          coord_face.data());
            
            for (size_t j = 0; j < face_to_full.size(); ++j) {
                coord_full[face_to_full[j]] = coord_face[j] + 1;
            }
            
            // Halo negativo
            coord_full[d] = 0;
            size_t idx_halo_minus = coord_to_index(N_dim, local_L_halo.data(), 
                                                   coord_full.data());
            conf_local[idx_halo_minus] = buffers.recv_minus[d][i];
            
            // Halo positivo
            coord_full[d] = local_L[d] + 1;
            size_t idx_halo_plus = coord_to_index(N_dim, local_L_halo.data(), 
                                                  coord_full.data());
            conf_local[idx_halo_plus] = buffers.recv_plus[d][i];
        }
    }
}

// halo_index: trova i rank vicini lungo ogni dimensione MPI
inline void halo_index(MPI_Comm cart_comm, int ndims, std::vector<std::vector<int>>& neighbors) {
    neighbors.resize(ndims, std::vector<int>(2));

    for (int d = 0; d < ndims; ++d) {
        int rank_source, rank_dest;
        MPI_Cart_shift(cart_comm, d, 1, &rank_source, &rank_dest);
        neighbors[d][0] = rank_source;
        neighbors[d][1] = rank_dest;
    }
}
