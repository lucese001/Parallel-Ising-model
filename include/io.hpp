#pragma once
#include <cstdio>
#include <cstddef>
#include <vector>
#include <mpi.h>

using std::vector;

// Legge i parametri di simulazione dal file dimensioni.txt
// Restituisce true se la lettura è andata a buon fine, false altrimenti
inline bool read_input_file(const char* filename,
                            size_t& N_dim,
                            vector<size_t>& arr,
                            size_t& nConfs,
                            size_t& nThreads,
                            double& Beta,size_t& seed) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Errore apertura %s: ", filename);
        perror("");
        return false;
    }

    if (fscanf(fp, "%zu", &N_dim) != 1) {
        fprintf(stderr, "Errore lettura N_dim\n");
        fclose(fp);
        return false;
    }   
    arr.resize(N_dim);

    for (size_t i = 0; i < N_dim; ++i) {
        if (fscanf(fp, "%zu", &arr[i]) != 1) {
            fprintf(stderr, "Errore lettura arr[%zu]\n", i);
            fclose(fp);
            return false;
        }
    }
    
    if (fscanf(fp, "%d", &nConfs) != 1) {
        fprintf(stderr, "Errore lettura nConfs\n");
        fclose(fp);
        return false;
    }
    
    if (fscanf(fp, "%zu", &nThreads) != 1) {
        fprintf(stderr, "Errore lettura nThreads\n");
        fclose(fp);
        return false;
    }
    
    if (fscanf(fp, "%lf", &Beta) != 1) {
        fprintf(stderr, "Errore lettura Beta\n");
        fclose(fp);
        return false;
    }
    
    if (fscanf(fp, "%zu", &seed) != 1) {
        fprintf(stderr, "Errore lettura seed\n");
        fclose(fp);
        return false;
    }
    
    fclose(fp);
    
    // Stampa ció che hai letto in caso (debug)
    printf("Rank 0 ha letto: N_dim=%zu, nConfs=%d, nThreads=%zu, Beta=%lg, seed=%zu\n", 
           N_dim, nConfs, nThreads, Beta, seed);
    printf("Dimensioni: ");
    for (size_t i = 0; i < N_dim; ++i) printf("%zu ", arr[i]);
    printf("\n");
    
    return true;
}

// Stampa il riepilogo delle prestazioni alla fine della simulazione
inline void print_performance_summary(double total, double compute, 
                                       double mpi, double io, double init,
                                       int nConfs) {
    double overhead = total - compute - mpi - io;
    printf("\n");
    printf("          PERFORMANCE PROFILING        \n");
    printf("Total runtime:             %10.3f s (100.0%%)\n", total);
    printf("Computation time:          %10.3f s (%5.1f%%)\n", compute, 100.0*compute/total);
    printf("MPI Communication:         %10.3f s (%5.1f%%)\n", mpi, 100.0*mpi/total);
    printf("I/O (file write):          %10.3f s (%5.1f%%)\n", io, 100.0*io/total);
    printf("Initialitation time:       %10.3f s (%5.1f%%)\n", init, 100.0*init/total);
    printf("Overhead:                  %10.3f s (%5.1f%%)\n", overhead, 100.0*overhead/total);
    printf("Configurations:               %d\n", nConfs);
    printf("Time per config:           %10.3f s\n", total/nConfs);
}

// Scrive una misura nel file di output
inline void write_measurement(FILE* measFile, double mag, double en, size_t N) {
    fprintf(measFile, "%lg %lg\n", mag/N, en/N);
    fflush(measFile);
}

// Stampa il progresso della simulazione
inline void print_progress(int iConf, int local_Nconfs, int nConfs, int world_size) {
    if ((iConf + 1) % 10 == 0 || (iConf + 1) == local_Nconfs) {
        int global_progress = 0;
        for (int r = 0; r < world_size; ++r) {
            int r_done = std::min(iConf + 1, 
                           (int)(nConfs / world_size + (r < (nConfs % world_size) ? 1 : 0)));
            global_progress += r_done;
        }

        printf("Progress: %d/%d (%.1f%%)\n", 
               global_progress, nConfs, 100.0 * global_progress / nConfs);
    }
}

// Stampa le informazioni sulla simulazione
inline void print_simulation_info(size_t N_dim, size_t N, size_t nThreads, int nConfs, 
                                   double Beta, size_t rng_memory, bool parallel_rng) {
    printf("N_dim: %zu, Npunti: %zu, NThreads: %zu, nConfs: %d, Beta: %lg\n", 
           N_dim, N, nThreads, nConfs, Beta);
    if (parallel_rng) {
        printf("Memory usage of the rng: %zu Bytes\n", rng_memory);
    } else {
        printf("Memory usage of the rng: %zu MB\n", rng_memory / (1 << 20));
    }
}

// Stampa la topologia MPI per debug
inline void print_mpi_topology(int world_rank, int world_size, size_t N_dim,
                                const vector<int>& rank_coords,
                                const vector<size_t>& global_offset,
                                const vector<size_t>& local_L) {
    for (int r = 0; r < world_size; ++r) {
        if (world_rank == r) {
            printf("RANK %d \n", world_rank);
            printf("  rank_coords: [");
            for (size_t d = 0; d < N_dim; ++d) {
                printf("%d", rank_coords[d]);
                if (d < N_dim-1) printf(", ");
            }
            printf("]\n");
        
            printf("  global_offset: [");
            for (size_t d = 0; d < N_dim; ++d) {
                printf("%zu", global_offset[d]);
                if (d < N_dim-1) printf(", ");
            }
            printf("]\n");

            printf("  local_L: [");
            for (size_t d = 0; d < N_dim; ++d) {
                printf("%zu", local_L[d]);
                if (d < N_dim-1) printf(", ");
            }
            printf("]\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
