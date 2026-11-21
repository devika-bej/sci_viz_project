#pragma once
#include <omp.h>
#include <vector>
#include "grid.h"
#include "order.h"

inline std::vector<int> compressPath(const OrderField &o, const Grid &grid, int *rounds = nullptr)
{
    std::vector<int> dm(grid.n);

#pragma omp parallel for
    for (int v = 0; v < grid.n; ++v)
    {
        int best = v;
        int neighbours[NB];
        grid.neighbor(v, neighbours);
        for (int i = 0; i < NB; ++i)
        {
            int u = neighbours[i];
            if (u != -1 && higher(o, u, best))
            {
                best = u;
            }
        }
        dm[v] = best;
    }

    int maxRounds = 0;
#pragma omp parallel reduction(max : maxRounds)
    {
        int t = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        const int begin = (int)((long long)grid.n * t / nthreads);
        const int end = (int)((long long)grid.n * (t + 1) / nthreads);

        std::vector<int> current, next;
        for (int v = begin; v < end; ++v)
        {
            if (dm[v] != v)
                current.push_back(v);
        }

        int r = 0;
        while (!current.empty())
        {
            ++r;
            next.clear();
            for (int v : current)
            {
                int i = dm[v], j;
#pragma omp atomic read
                j = dm[i];
                if (i != j)
                {
#pragma omp atomic write
                    dm[v] = j;
                    next.push_back(v);
                }
            }
            current.swap(next);
        }
        maxRounds = std::max(maxRounds, r);
    }
    if (rounds)
        *rounds = maxRounds;
    return dm;
}