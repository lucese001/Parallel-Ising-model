#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>


#ifdef USE_PHILOX
#include "philox_rng.hpp"
#else
#include "prng_engine.hpp"
#endif

#include "utility.hpp"
#include "ising.hpp"

using namespace std;

// Variabili globali esterne (definite in main.cpp)
extern int N_dim;
extern long long N;
extern double Beta;

// metropolis_update: esegue uno sweep Metropolis
// con aggiornamento a scacchiera sui siti specificati
// sites[] contiene gli indici HALO (pronti per accedere a conf_local)

#ifdef USE_PHILOX

// Philox RNG: riproducibile per update Bulk-Boundary (non dipende dalla
// sequenza estratta).
void metropolis_update(vector<int8_t>& conf_local,
                              const vector<size_t>& sites,
                              const vector<size_t>& sites_global_indices,
                              const vector<size_t>& stride_halo,
                              const vector<double>& expTable,
                              long long &DeltaE,
                              long long &DeltaMag,
                              PhiloxRNG& gen,
                              int iConf,
                              size_t nThreads)
{
        long long local_dE = 0; // Variazione di energia locale a ogni thread
        long long local_dM = 0; // Variazione di magnetizzazione locale a ogni thread
#pragma omp parallel reduction(+:local_dE, local_dM)

    {
        const size_t iThread = omp_get_thread_num();
        const size_t chunkSize = (sites.size() + nThreads - 1) / nThreads;
        const size_t beg = chunkSize * iThread;
        const size_t end = std::min(sites.size(), beg + chunkSize);

        for (size_t idx = beg; idx < end; ++idx) {
            const size_t iSite_halo = sites[idx];
            const size_t global_idx = sites_global_indices[idx];

            const int8_t oldVal = conf_local[iSite_halo];

            // Philox e' un counter-based RNG. L'ordine in cui
            // vengono aggiornati i siti non influenza i numeri estratti
            // (dipendono solo dall'indice globale e dalla configurazione)

            // Proposta di spin: se é uguale a quello precedente
            // salta al prossimo sito
            uint32_t rand0 = gen.get1(global_idx, iConf, 0, false);
            int8_t proposed_spin = (rand0 & 1) ? 1 : -1;
            if (proposed_spin == oldVal) continue;

            const int enBefore = computeEnSite(conf_local, iSite_halo,
                                               stride_halo, N_dim);
            const int eDiff = - 2*enBefore; // Se flippa l'energia cambia segno           
            
            // Accettazione
            bool accept;
            if (eDiff <= 0) {
                // L'energia diminuisce quindi il flip viene accettato
                accept = true;
            } else {
                // eDiff > 0: valori possibili 4, 8, ..., 4*N_dim
                // Lookup: expTable[eDiff/4 - 1] = exp(-Beta*eDiff)
                uint32_t rand1 = gen.get1(global_idx, iConf, 1, false);
                // Questa parte non mi è 100% chiara
                const double rand_uniform = (double)rand1 / 4294967296.0;
                accept = (rand_uniform < expTable[eDiff / 4 - 1]);
            }

            if (accept) {
                conf_local[iSite_halo] = proposed_spin;
                local_dE += eDiff;
                local_dM += proposed_spin - oldVal;
            }

            #ifdef DEBUG_PRINT
            #pragma omp critical
            {
                std::cout << "PHILOX_DEBUG: iConf=" << iConf
                << " global_idx=" << global_idx
                << " rand0=" << rand0
                << " spin=" << (int)proposed_spin
                << " E iniziale=" << enBefore
                << " DeltaE=" << eDiff
                << " acc=" << accept
                << " oldVal=" << +oldVal
                << " proposed_spin=" << +proposed_spin
                << " conf_finale=" << +conf_local[iSite_halo] << "\n";
            }
            #endif
        }
    }
    // Somma delle energie per thread
    DeltaE += local_dE;
    DeltaMag += local_dM;
}

#else  // Versione originale prng_engine (dipende dalla sequenza estratta)

inline void metropolis_update(vector<int8_t>& conf_local,
                              const vector<size_t>& sites,
                              const vector<size_t>& sites_global_indices,
                              const vector<size_t>& stride_halo,
                              prng_engine& gen,
                              int iConf,
                              size_t nThreads)
{
    #pragma omp parallel
    {
        const size_t iThread = omp_get_thread_num();
        const size_t chunkSize = (sites.size() + nThreads - 1) / nThreads;
        const size_t beg = chunkSize * iThread;
        const size_t end = std::min(sites.size(), beg + chunkSize);

        for (size_t idx = beg; idx < end; ++idx) {
            const size_t iSite_halo = sites[idx];
            const size_t global_idx = sites_global_indices[idx];

            // discard basato sull'indice globale
            prng_engine genView = gen;
            genView.discard(2 * 2 * (global_idx + N * iConf));

            const int8_t oldVal = conf_local[iSite_halo];
            const int enBefore = computeEnSite(conf_local, iSite_halo,
                                               stride_halo, N_dim);

            conf_local[iSite_halo] = (int8_t)(binomial_distribution<int>(1, 0.5)(genView) * 2 - 1);

            const int enAfter = computeEnSite(conf_local, iSite_halo,
                                              stride_halo, N_dim);
            const int eDiff = enAfter - enBefore;
            const double pAcc = std::min(1.0, exp(-Beta * (double)eDiff));
            const int acc = binomial_distribution<int>(1, pAcc)(genView);

            if (!acc) conf_local[iSite_halo] = oldVal;
        }
    }
}

#endif
