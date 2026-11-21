#pragma once
#include <cstdint>
#include "grid.h"
#include "order.h"

struct LinkTables
{
    uint16_t adjMask[NB];
    uint8_t lut[1 << NB];
};

struct CriticalPoints
{
    std::vector<int> maxima;
    std::vector<SaddleInfo> saddles;
};

inline bool isOffset(int dx, int dy, int dz)
{
    for (int i = 0; i < NB; ++i)
    {
        if (dx == OFFSETS[i][0] && dy == OFFSETS[i][1] && dz == OFFSETS[i][2])
        {
            return true;
        }
    }
    return false;
}

inline void buildAdjacencyMask(LinkTables &lt)
{
    for (int i = 0; i < NB; ++i)
    {
        uint16_t mask = 0;
        for (int j = 0; j < NB; ++j)
        {
            if (i == j)
            {
                continue;
            }
            if (isOffset(OFFSETS[j][0] - OFFSETS[i][0], OFFSETS[j][1] - OFFSETS[i][1], OFFSETS[j][2] - OFFSETS[i][2]))
            {
                mask |= (1 << j);
            }
        }
        lt.adjMask[i] = mask;
    }
}

inline int countComponents(uint16_t mask, const uint16_t adjMask[NB])
{
    int k = 0;
    while (mask)
    {
        uint16_t comp = mask & (uint16_t)(-mask);
        uint16_t prev = 0;
        while (comp != prev)
        {
            prev = comp;
            for (int i = 0; i < NB; ++i)
            {
                if (comp & (1u << i))
                {
                    comp |= (adjMask[i] & mask);
                }
            }
        }
        mask &= ~comp;
        ++k;
    }
    return k;
}

inline void buildLUT(LinkTables &lt)
{
    buildAdjacencyMask(lt);
    for (uint16_t i = 0; i < (1 << NB); ++i)
    {
        lt.lut[i] = countComponents(i, lt.adjMask);
    }
}

inline CriticalPoints classifyCriticalPoints(const Grid &grid, const OrderField &o, const std::vector<int> &dm, const LinkTables &lt, bool useLUT)
{
    CriticalPoints cp;
#pragma omp parallel
    {
        std::vector<int> localMaxima;
        std::vector<SaddleInfo> localSaddles;

#pragma omp for schedule(static)
        for (int v = 0; v < grid.n; ++v)
        {
            int neighbors[NB];
            grid.neighbor(v, neighbors);
            int mask = 0, first = -1, distinct = 0;
            for (int i = 0; i < NB; ++i)
            {
                int u = neighbors[i];
                if (u != -1 && o.order[u] > o.order[v])
                {
                    mask |= (1 << i);
                    if (first == -1)
                    {
                        first = dm[u];
                    }
                    else
                    {
                        if (dm[u] != first)
                        {
                            distinct = 1;
                        }
                    }
                }
            }

            if (mask == 0)
            {
                localMaxima.push_back(v);
            }
            else if (distinct == 1)
            {
                int k = useLUT ? lt.lut[mask] : countComponents(mask, lt.adjMask);
                if (k >= 2)
                {
                    localSaddles.push_back({v, k - 1});
                }
            }
        }

#pragma omp critical
        {
            cp.maxima.insert(cp.maxima.end(), localMaxima.begin(), localMaxima.end());
            cp.saddles.insert(cp.saddles.end(), localSaddles.begin(), localSaddles.end());
        }
    }

    return cp;
}