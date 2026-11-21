#pragma once
#include <vector>
#include <numeric>
#include "extremum_graph.h"

struct GraphSplitTree
{
    const ExtremumGraph &g;
    const std::vector<int> &order, &sorted;
    std::vector<int> parent;
    std::vector<PersistencePair> pairs;

    GraphSplitTree(const ExtremumGraph &g, const OrderField &o)
        : g(g), order(o.order), sorted(o.sorted), parent(o.order.size())
    {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int v)
    {
        while (parent[v] != v)
        {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        return v;
    }

    int unite(int u, int v)
    {
        int rootU = find(u);
        int rootV = find(v);
        if (rootU != rootV)
        {
            if (order[rootU] > order[rootV])
            {
                parent[rootV] = rootU;
                return rootU;
            }
            else
            {
                parent[rootU] = rootV;
                return rootV;
            }
        }
        return rootU;
    }

    void compute()
    {
        const int S = (int)g.saddles.size();
        for (int i = 0; i < S; ++i) // saddles already sorted by order DESC
        {
            const int s = g.saddles[i];
            int roots[NB], k = 0;
            for (int e = g.offset[i]; e < g.offset[i + 1]; ++e)
            {
                const int r = find(g.nbr[e]);
                bool seen = false;
                for (int j = 0; j < k; ++j)
                    if (roots[j] == r)
                    {
                        seen = true;
                        break;
                    }
                if (!seen)
                    roots[k++] = r;
            }
            if (k >= 2) // real merge
            {
                int winner = roots[0];
                for (int j = 1; j < k; ++j)
                    if (order[roots[j]] > order[winner])
                        winner = roots[j];
                for (int j = 0; j < k; ++j)
                    if (roots[j] != winner)
                        pairs.push_back({roots[j], s, winner});
            }
            for (int j = 0; j < k; ++j)
                unite(s, roots[j]); // k == 1: loop saddle, nothing new
        }
        const int n = (int)sorted.size();
        pairs.push_back({sorted[n - 1], sorted[0], -1});
    }
};