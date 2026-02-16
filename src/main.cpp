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

int world_rank; // Rank (id) del processo
int world_size; // Numero di rank

// Parametri della simulazione
long long N=1;       // Numero totale di siti
double Beta;         // Inverso della temperatura
int nThreads;        // Numero di thread OpenMP
int N_dim;           // Numero di dimensioni
vector<size_t> arr;  // Lunghezze del reticolo per dimensione
int nConfs;          // Numero di configurazioni
size_t seed;         // Seed per il generatore di numeri casuali

// Definizione della variabile statica timerCost (dichiarata in utility.hpp)
timer timer::timerCost;

int main(int argc, char** argv) {

    timer totalTime, computeTime, mpiTime, ioTime,setupTime, div_time;
    totalTime.start();
    setupTime.start();

    // Sottrae il costo del timer alla simulazione
    for (size_t i = 0; i < 100000; ++i) { 
        timer::timerCost.start(); 
        timer::timerCost.stop(); 
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        
    // Lettura parametri: da riga di comando o da file
    // Uso: ./ising_philox.exe <N_dim> <L0> [L1 ...] <nConfs> <nThreads> <Beta> <seed>
    if (world_rank == 0) {
        if (argc > 1) {
            N_dim = atoi(argv[1]);
            int expected = N_dim + 6;
            if (argc != expected) {
                fprintf(stderr, "Uso: %s <N_dim> <L0> [L1 ...] <nConfs> <nThreads> <Beta> <seed>\n", argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            arr.resize(N_dim);
            for (int d = 0; d < N_dim; ++d)
                arr[d] = (size_t)atoll(argv[2 + d]);
            nConfs = atoi(argv[2 + N_dim]);
            nThreads = atoi(argv[3 + N_dim]);
            Beta = atof(argv[4 + N_dim]);
            seed = (size_t)atoll(argv[5 + N_dim]);
        } else {
            if (!read_input_file("input/dimensioni.txt", N_dim,
                arr, nConfs, nThreads, Beta, seed)) {
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }
    
    //Broadcast dati agli altri processi
    MPI_Bcast(&N_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        arr.resize(N_dim);
    }
    MPI_Bcast(arr.data(), N_dim*sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nConfs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nThreads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    omp_set_num_threads((int)nThreads);
    
    vector<int> Chunks(N_dim); //Numero di processi allocati lungo ogni dimensione
    vector<size_t> local_L(N_dim); //Dimensioni locali del nodo
    vector<int> rank_coords(N_dim); // Coordinate cartesiane del rank nella griglia MPI
    MPI_Comm cart_comm;              // Comunicatore cartesiano
    vector<int> periods(N_dim, 1);  // Condizioni periodiche ai bordi
    
    MPI_Dims_create(world_size, N_dim, Chunks.data());
    MPI_Cart_create(MPI_COMM_WORLD,(int)N_dim,Chunks.data(),
                    periods.data(),1,&cart_comm);
    MPI_Cart_coords(cart_comm, world_rank, N_dim, 
                    rank_coords.data());

    //Vettore che contiene, per ogni sito al confine, il rank dei 
    //processi MPI dei suoi vicini lungo ogni dimensione.
    std::vector<std::vector<int>> neighbors;
    halo_index(cart_comm, N_dim, neighbors);

    size_t N_local = 1; // Numero di siti del nodo
    size_t N_alloc = 1; // Numero di siti del nodo + halo
    vector<size_t> local_L_halo(N_dim); //Lato del rank + halo
    vector<size_t> global_offset(N_dim); //Offset per passare da coordinate 
                                         //del rank a coordinate globali

    // Si controlla che il reticolo sia divisibile in Chunks uguali
    for (int d = 0; d < N_dim; ++d) {
        if (arr[d] % Chunks[d] != 0) {
            if (world_rank == 0){
                cerr << "Errore: arr[" << d << "] non divisibile per" 
                <<"Chunks uguali. Prova un'altra combinazione di rank"
                <<"e lati[" << d << "]\n";
                }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        local_L[d] = arr[d] / Chunks[d];
        local_L_halo[d] = local_L[d] + 2;
        N_local *= local_L[d]; 
        N *= arr[d];
        global_offset[d] = rank_coords[d] * local_L[d];
        // local_L_halo padded: usato affinché ogni riga abbia un numero
        // intero di word (ho un numero di bit multiplo di 64).
    }

    // Padding del reticolo halo lungo la direzione 0. Questo é utile per il
    // loop metropolis, che itera lungo le righe in direzione 0: in questo modo
    // i siti (bit) su una riga appartengono a word diverse. Dato che ogni thread
    // itera su righe diverse, evitiamo race condition di accesso ai bit.
    local_L_halo[0]=((local_L_halo[0] + 63) / 64) * 64;

    // Numero di bit per ogni rank (tiene conto dell'halo e del padding)
    N_alloc = 1;
    for (int d = 0; d < N_dim; ++d){
      N_alloc *= local_L_halo[d];  
    }

    //Precalcola gli stride locali a ogni rank
    vector<uint32_t> stride_halo(N_dim);
    stride_halo[0] = 1;
    for (int d = 1; d < N_dim; ++d){
        stride_halo[d] = stride_halo[d-1] * local_L_halo[d-1];
    }
    // Calcola l'indice oltre il quale non fare prefetch (rischia di sforare)
#ifdef PREFETCH_CACHE
    size_t pf_limit = N_alloc - stride_halo[N_dim - 1];
#else
    size_t pf_limit = 0; // non usato, serve per compilare la firma
#endif
    //Precalcola gli stride globali
    vector<long long> stride_global(N_dim);
    stride_global[0] = 1;
    for (int d = 1; d < N_dim; ++d){
        stride_global[d] = stride_global[d-1] * arr[d-1];
    }

    // Vettori per siti al confine (Red/Black), usati da entrambi i path
    vector<uint32_t> boundary_sites[2];
    vector<uint32_t> boundary_indices[2];

    #ifdef IDX_ALLOC
        // Path IDX_ALLOC: anche i bulk sono pre-allocati
        vector<uint32_t> bulk_sites[2];
        vector<uint32_t> bulk_indices[2];
        classify_bulk(N_dim, local_L, local_L_halo,
              global_offset, arr, bulk_sites, bulk_indices);
    #endif

    //Inizializzazione RNG
    // Philox counter-based: il seed seleziona l'esperimento,
    // il counter (global_idx, iConf) seleziona il numero random
    uint32_t rng_seed = (uint32_t)(seed + 104729);
    print_simulation_info(N_dim, N, nThreads, nConfs, Beta, world_size
                          sizeof(uint32_t), true);
    // Vettore che contiene la configurazione locale a ogni rank
    // (1 byte per sito). Contiene celle halo
    vector<uint64_t> conf_local ((N_alloc+63)/64); 

    //Genera la prima configurazione
    initialize_configuration(conf_local, N_local, N_dim, local_L,
                            local_L_halo,global_offset, arr, rng_seed);

    // Costruisce le dimensioni delle facce e la sua posizione
    vector<FaceInfo> faces = build_faces(local_L, N_dim);
    // Salva gli indici (locali e globali) dei siti che appartengono 
    // a ogni faccia.
    vector<FaceCache> face_cache = build_face_cache(faces, local_L,
                                                    local_L_halo,
                                                    global_offset,
                                                    arr, N_dim,
                                                    boundary_sites,
                                                    boundary_indices);

    //static int global_conf_count = 0; //Numero di configurazioni globali
    vector<MPI_Request> requests; //definizione richieste processi MPI
    HaloBuffers buffers; //definizione buffers
    buffers.resize(N_dim);

    //Halo exchange per calcolo energia e magnetizzazione iniziale
    start_halo_exchange(conf_local, local_L, local_L_halo,
                        neighbors, cart_comm, N_dim, buffers,
                        faces, requests, face_cache, 0, true);
    write_halo_data(conf_local, buffers, faces, local_L,
                    local_L_halo, N_dim, face_cache, 0, requests);
    start_halo_exchange(conf_local, local_L, local_L_halo,
                        neighbors, cart_comm, N_dim, buffers,
                        faces, requests, face_cache, 1, true);
    write_halo_data(conf_local, buffers, faces, local_L,
                    local_L_halo, N_dim, face_cache, 1, requests);
    
    long long E_rank = computeEn_rank(conf_local, stride_halo, 
                                        local_L, N_dim);
    long long Mag_rank = compute_Mag_rank(conf_local, stride_halo, 
                                            local_L, N_dim);

    // Riduzione per calcolo di energia e magnetizzazione iniziale (globale)
    long long E = 0;
    long long Mag = 0;
    MPI_Reduce(&E_rank, &E, 1, MPI_LONG_LONG, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&Mag_rank, &Mag, 1, MPI_LONG_LONG, MPI_SUM, 0, cart_comm);

    // Lookup table per precalcolare l'esponenziale usata in Metropolis
    // Le differenze di energia positive non 0 possibili sono N_dim
    // e sono multipli di 4, dato che le energie possibili sono multipli
    // di 2 e eDiff=-2*E (se flippa). Esempi di differenze di energia 
    // 2D=[4,8] 3D=[4,8,12]

    vector<double> expTable(N_dim);
    for (int d = 0; d < N_dim; ++d){
        expTable[d] = exp(-Beta * 4.0 * (d + 1));
    }

    // Apertura del file di output per le misure
    FILE* measFile = nullptr;
    if (world_rank == 0) {
        string fname = "output/meas_" + to_string(world_size) + "rank";
        for (int d = 0; d < N_dim; ++d)
            fname += (d == 0 ? "_" : "x") + to_string(arr[d]);
        fname += ".txt";

        measFile = fopen(fname.c_str(), "w");
        if (!measFile) {
            perror(("Errore apertura " + fname).c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        master_printf("Output file: %s\n", fname.c_str());
    }

    // Debug per vedere la topologia MPI
    #ifdef DEBUG_PRINT
        print_mpi_topology (world_rank, world_size, N_dim, 
                            rank_coords, global_offset, local_L);
    #endif

    setupTime.stop();

    for (int iConf = 0; iConf < nConfs; ++iConf) {

        //Stampa debug
        #ifdef DEBUG_PRINT
            // Stampa la configurazione globale per debug (utile 
            // per verificare riproducibilità)
            print_global_configuration_debug(conf_local, local_L, local_L_halo, global_offset, arr,
                                             N_dim, N_local, N, world_rank, world_size, 
                                             iConf, cart_comm);
        #endif

        // Aggiornamento Rosso/Nero.
        // L'ordine dei processi é il seguente:
        //1)avvia la comunicazione MPI per gli halo neri
        //2)update Bulk rosso
        //3)aspetta la sincronizzazione e scrive gli halo neri
        //4)update Boundary rosso
        //5)avvia la comunicazione MPI per gli halo rossi
        //6)update Bulk nero
        //7)aspetta la sincronizzazione e scrive gli halo neri
        //8)update Boundary nero
        
        // Variazione di energia e magnetizzazione locali a ogni rank
        long long DeltaE = 0;
        long long DeltaMag = 0;


        #ifdef IDX_ALLOC

	        for(int updPar=0;updPar<2;updPar++){

	            const int commPar=1-updPar;
	    
	            // Inizia l' halo exchange nero/rosso
                mpiTime.start();
	            start_halo_exchange(conf_local, local_L, 
                                local_L_halo, neighbors, 
                                cart_comm, N_dim, buffers, 
                                faces, requests, 
                                face_cache,commPar, true);
	            mpiTime.stop();
	    
	            computeTime.start();
	             // Update Bulk rosso/nero
	            metropolis_update(conf_local, bulk_sites[updPar],
			                  bulk_indices[updPar], stride_halo, 
                              expTable, DeltaE,DeltaMag, rng_seed,
                              iConf, nThreads);

	            computeTime.stop();
	            mpiTime.start();
	            // Scrivi gli halo
	            write_halo_data(conf_local, buffers, faces, 
                            local_L, local_L_halo, N_dim, 
                            face_cache, commPar, requests);
	            mpiTime.stop();
	            computeTime.start();
	            // Update boundary rossa/nero
	            metropolis_update(conf_local, boundary_sites[updPar],
			                  boundary_indices[updPar], stride_halo,
                              expTable, DeltaE,DeltaMag, rng_seed,
                              iConf, nThreads);
                computeTime.stop();
            } //Fine loop sulle paritá

        #endif //IDX_ALLOC

        #ifdef ROWING

            for(int updPar=0;updPar<2;updPar++){

        	    const int commPar=1-updPar;
	            // Inizia l' halo exchange nero/rosso
                mpiTime.start();
	            start_halo_exchange(conf_local, local_L, 
                                local_L_halo, neighbors, 
                                cart_comm, N_dim, buffers, 
                                faces, requests, 
                                face_cache,commPar, true);
	            mpiTime.stop();
	    
	            computeTime.start();
	            div_time.start();
	             // Update Bulk rosso/nero (include divisioni per righe)
                metropolis_update_bulk(conf_local,updPar,
                                        local_L, local_L_halo,
                                        global_offset, arr,
                                        stride_halo, stride_global, expTable,
                                        pf_limit, DeltaE, DeltaMag,
                                        rng_seed, iConf);
	            div_time.stop();
	            computeTime.stop();
	            mpiTime.start();

	            // Scrivi gli halo
	            write_halo_data(conf_local, buffers, faces, 
                            local_L, local_L_halo, N_dim, 
                            face_cache, commPar, requests);
	            mpiTime.stop();
	            computeTime.start();
	            // Update boundary rossa/nero
	            metropolis_update(conf_local, boundary_sites[updPar],
			                  boundary_indices[updPar], stride_halo,
                              expTable, DeltaE,DeltaMag, rng_seed,
                              iConf, nThreads);
                computeTime.stop();
            } //Fine loop sulle paritá

        #endif //ROWING
            

        // Somma delle variazion locali a ogni rank per ottenere
        // le variazioni di energia e magnetizzazione globali
        mpiTime.start();
        long long DeltaE_glob = 0;
        long long DeltaMag_glob = 0;
        MPI_Reduce(&DeltaE, &DeltaE_glob, 1, MPI_LONG_LONG, MPI_SUM, 0, cart_comm);
        MPI_Reduce(&DeltaMag, &DeltaMag_glob, 1, MPI_LONG_LONG, MPI_SUM, 0, cart_comm);
        mpiTime.stop();
        
        // Si scrivono le misure nel file
        if (world_rank == 0) {
            ioTime.start();
            E += DeltaE_glob;
            Mag += DeltaMag_glob;
            write_measurement(measFile, Mag, E, N);
            /*print_progress(iConf, local_Nconfs, nConfs, world_size);*/
            ioTime.stop();
        }
    } // Fine del loop sulle configurazioni
    
    if (world_rank == 0 && measFile) {
        fclose(measFile);
    }

    totalTime.stop();
    
    if (world_rank == 0) {
        print_performance_summary(totalTime.get(), computeTime.get(),
                                  mpiTime.get(), ioTime.get(),
                                  setupTime.get(), nConfs);
        #ifdef ROWING
        master_printf("Bulk (rowing) time: %.6f s (%.6f s/conf) | "
                      "Boundary time: %.6f s (%.6f s/conf)\n",
                      div_time.get(), div_time.get() / nConfs,
                      computeTime.get() - div_time.get(),
                      (computeTime.get() - div_time.get()) / nConfs);
        #endif
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
