#pragma once
#include <vector>
#include <algorithm>
#include "critical_points.h"

struct ExtremumGraph
{
    std::vector<int> maxima, saddles, offset, nbr;
};

inline ExtremumGraph buildGraph(const Grid &grid, const OrderField &o, const std::vector<int> &dm, const CriticalPoints &cp)
{
    ExtremumGraph g;
    g.maxima = cp.maxima;
    for (size_t i = 0; i < cp.saddles.size(); ++i)
    {
        g.saddles.push_back(cp.saddles[i].v);
    }
    std::sort(g.saddles.begin(), g.saddles.end(), [&](int a, int b)
              { return o.order[a] > o.order[b]; });
    int numSaddles = g.saddles.size();

    std::vector<std::vector<int>> tmp(numSaddles, std::vector<int>(NB, -1));
    std::vector<int> cnt(numSaddles, 0);
#pragma omp parallel for
    for (int i = 0; i < numSaddles; ++i)
    {
        int v = g.saddles[i];
        int c = 0;
        int neigh[NB];
        grid.neighbor(v, neigh);
        for (int u : neigh)
        {
            if (u != -1 && o.order[u] > o.order[v])
            {
                int m = dm[u];
                for (int j = 0; j < c; ++j)
                {
                    if (tmp[i][j] == m)
                    {
                        m = -1;
                        break;
                    }
                }
                if (m != -1)
                {
                    tmp[i][c++] = m;
                }
            }
        }
        cnt[i] = c;
    }

    g.offset.push_back(0);
    for (int i = 0; i < numSaddles; ++i)
        g.offset.push_back(g.offset.back() + cnt[i]);

    g.nbr.resize(g.offset.back());
#pragma omp parallel for
    for (int i = 0; i < numSaddles; ++i)
    {
        for (int j = 0; j < cnt[i]; ++j)
            g.nbr[g.offset[i] + j] = tmp[i][j];
    }

    return g;
}

inline long long gridEdgeCount(const Grid &grid)
{
    long long e = 0;
    for (int i = 0; i < NB; ++i)
    {
        const int dx = OFFSETS[i][0], dy = OFFSETS[i][1], dz = OFFSETS[i][2];
        if (dx < 0 || dy < 0 || dz < 0)
            continue;
        const long long a = grid.nx - dx, b = grid.ny - dy, c = grid.nz - dz;
        if (a > 0 && b > 0 && c > 0)
            e += a * b * c;
    }
    return e;
}