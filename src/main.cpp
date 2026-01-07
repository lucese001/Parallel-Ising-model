#define PARALLEL_RNG

#include "prng_engine.hpp"
#include "utility.hpp"
#include "ising.hpp"
#include "metropolis.hpp"
#include "halo.hpp"
#include "io.hpp"

#include <cstdint>
#include <random>
#include <vector>
#include <cstdio>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>

using namespace std;

// Parametri della simulazione
size_t N = 1;              // numero totale di siti
double Beta;               // inverso della temperatura
size_t nThreads;           // numero di thread OpenMP
size_t N_dim;              // numero di dimensioni
vector<size_t> arr;        // lunghezze del reticolo per dimensione
size_t nConfs;             // numero di configurazioni
size_t seed;               // seed per il generatore di numeri casuali

// Definizione della variabile statica timerCost (dichiarata in utility.hpp)
timer timer::timerCost;

int main(int argc, char** argv) {
    timer totalTime, computeTime, mpiTime, ioTime,setupTime;
    totalTime.start();
    setupTime.start();
    int world_size; // Numero di processi
    int world_rank; // Rank (id) del processo
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("world_size  %d rank %d\n", world_size, world_rank);
    
    // Lettura del file di input
    if (world_rank == 0) {
        if (!read_input_file("input/dimensioni.txt", N_dim, arr, nConfs, nThreads, Beta,seed)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    //Broadcast dati agli altri processi
    MPI_Bcast(&N_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        arr.resize(N_dim);
    }
    MPI_Bcast(arr.data(), N_dim, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nConfs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nThreads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //Calcolo del numero totale di siti (dopo aver ricevuto N_dim e arr)
    for (size_t i = 0; i < N_dim; ++i) {
        N *= arr[i];
    }
    if (world_rank != 0) {
        arr.resize(N_dim);
    }

    //Broadcast dati agli altri processi
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(arr.data(), N_dim, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nConfs, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nThreads, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    vector<int> Chunks(N_dim); //numero di processi allocati lungo ogni dimensione
    vector<size_t> local_L(N_dim); //dimensioni locali del nodo
    MPI_Dims_create(world_size, N_dim, Chunks.data());

    // Controllo che arr sia divisibile per Chunks
    for (size_t d = 0; d < N_dim; ++d) {
        if (arr[d] % Chunks[d] != 0) {
            if (world_rank == 0)
                cerr << "Errore: arr[" << d << "] non divisibile per Chunks uguali[" << d << "]\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        local_L[d] = arr[d] / Chunks[d];

    }

    size_t N_local = 1;
    size_t N_alloc = 1;
    vector<size_t> local_L_halo(N_dim);
    for (size_t d = 0; d < N_dim; ++d) {
        local_L_halo[d] = local_L[d] + 2;
        N_local *= local_L[d];      // numero di siti del nodo
        N_alloc *= local_L_halo[d]; // numero di siti del nodo + halo
    }

    vector<int> rank_coords(N_dim); // Coordinate cartesiane del rank nella griglia MPI
    MPI_Comm cart_comm;              // Comunicatore cartesiano
    vector<int> periods(N_dim, 1);  // Condizioni periodiche ai bordi
    MPI_Cart_create(MPI_COMM_WORLD,(int)N_dim,Chunks.data(),
                    periods.data(),1,&cart_comm);
    MPI_Cart_coords(cart_comm, world_rank, N_dim, 
                    rank_coords.data());

    //Calcolo dell'offset globale. Questo permette di trasformare
    // coordinate locali al rank in coordinate globali
    vector<size_t> global_offset(N_dim);
    for (size_t d = 0; d < N_dim; ++d) {
        global_offset[d] = rank_coords[d] * local_L[d];
    }

    // Debug per vedere la topologia MPI
    print_mpi_topology(world_rank, world_size, N_dim, 
                       rank_coords, global_offset, local_L);

    //Calcolo del numero di configurazioni locali
    size_t local_Nconfs;
    local_Nconfs = nConfs / world_size;
    if (world_rank < (nConfs % world_size)) {
        local_Nconfs++;
    }

    //Vettore che contiene, per ogni sito,
    // i suoi vicini lungo ogni dimensione.
    std::vector<std::vector<int>> neighbors; 
    halo_index(cart_comm, (int)N_dim, neighbors);
    omp_set_num_threads((int)nThreads);

    // Apertura del file di output per le misure
    FILE* measFile = nullptr;
    if (world_rank == 0) {
        measFile = fopen("output/meas.txt", "w");
        if (!measFile) {
            perror("Errore apertura output/meas.txt");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    for (size_t i = 0; i < 100000; ++i) { 
        timer::timerCost.start(); 
        timer::timerCost.stop(); 
    }

#ifdef PARALLEL_RNG
    //Inizializzazione del generatore di numeri casuali
    prng_engine gen(seed + world_rank * 104729);
    print_simulation_info(N_dim, N, nThreads, nConfs, Beta,
                          sizeof(prng_engine), true);
#endif
    vector<int> conf_local(N_alloc); //Vettore che contiene la configurazione locale
    vector<size_t> bulk_sites; //Vettore che contiene la configurazione dei siti bulk
    vector<size_t> boundary_sites; //Vettore che contiene la configurazione dei siti boundary
    vector<size_t> bulk_global_indices; //Vettore che contiene gli indici globali dei siti bulk
    vector<size_t> boundary_global_indices; //Vettore che contiene gli indici globali dei siti boundary

    //Genera la prima configurazione
    initialize_configuration(conf_local, N_alloc, gen);
    

    // Classificazione dei siti in bulk (siti con vicini all'
    // interno del nodo) e boundary (siti con vicini
    // fuori dal nodo)
    classify_sites(N_local, N_dim, local_L, global_offset, arr,
                   bulk_sites, bulk_global_indices,
                   boundary_sites, boundary_global_indices);

    static int global_conf_count = 0; //Numero di configurazioni globali
    vector<FaceInfo> faces = build_faces(local_L, N_dim); // Costruisce le informazioni delle facce per lo scambio halo
    vector<MPI_Request> requests; //definizione richieste processi MPI
    HaloBuffers buffers; //definizione buffers
    buffers.resize(N_dim);
    setupTime.stop();
    
    for (int iConf = 0; iConf < (int)local_Nconfs; ++iConf) {
        
        mpiTime.start();
        // Si iniziano a scambiare gli halo. La funzione non si blocca
        // ad aspettare che tutti gli scambi siano stati completati.
        start_halo_exchange(conf_local, local_L, local_L_halo, 
                           neighbors, cart_comm, N_dim, 
                           buffers, faces, requests);
        mpiTime.stop();
        if(world_rank == 0){
            for (size_t i = 0; i < N_local; ++i) {
                cout<<conf_local[i]<<" ";
            }
            cout<<endl;
        }
        if (world_rank==1){
            for (size_t i = 0; i < N_local; ++i) {
                cout<<conf_local[i]<<" ";
            }
            cout<<endl;
        }
        if (world_rank==2){
            for (size_t i = 0; i < N_local; ++i) {
                cout<<conf_local[i]<<" ";
            }
            cout<<endl;
        }
        // Si aggiornano i siti interni mentre MPI comunica
        computeTime.start();
        metropolis_update(conf_local, bulk_sites, 
                          bulk_global_indices,
                          local_L, local_L_halo, gen, 
                          iConf, nThreads, N_local);
        computeTime.stop();
        
        // Si aspetta il completamento della comunicazione MPI
        mpiTime.start();
        finish_halo_exchange(requests);
        // Si scrivono gli halo ricevuti in conf_local
        write_halo_data(conf_local, buffers, faces, local_L, local_L_halo, N_dim);
        mpiTime.stop();
        
        // Si aggiornano i siti al bordo
        computeTime.start();
        metropolis_update(conf_local, boundary_sites, 
                          boundary_global_indices,
                          local_L, local_L_halo, gen, 
                          iConf, nThreads, N_local);
        computeTime.stop();
        
        // Si misura la magnetizzazione e l'energia in ogni processo
        computeTime.start();
        double local_mag = computeMagnetization_local(conf_local, N_local, 
                                                      local_L, local_L_halo);
        double local_en = computeEn(conf_local, N_local, 
                                    local_L, local_L_halo);
        computeTime.stop();
        
        // Si calcolano la magnetizzazione e l'energia globali
        mpiTime.start();
        double global_mag, global_en;
        MPI_Reduce(&local_mag, &global_mag, world_size, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
        MPI_Reduce(&local_en, &global_en, world_size, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
        mpiTime.stop();
        
        // Si scrivono le misure nel file
        if (world_rank == 0) {
            ioTime.start();
            write_measurement(measFile, global_mag, global_en, N);
            print_progress(iConf, local_Nconfs, nConfs, world_size);
            ioTime.stop();
        }
    } // fine del loop sulle configurazioni
    
    if (world_rank == 0 && measFile) {
        fclose(measFile);
    }
    totalTime.stop();
    
    if (world_rank == 0) {
        print_performance_summary(totalTime.get(), computeTime.get(), 
                                  mpiTime.get(), ioTime.get(),setupTime.get(), nConfs);
    }
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}