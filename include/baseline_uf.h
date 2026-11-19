#pragma once
#include <vector>
#include <numeric>
#include "grid.h"
#include "order.h"

struct PersistencePair
{
    int maxV, saddle, parentMax;
};

struct SaddleInfo
{
    int v, multiplicity;
};

struct BaselineSplitTree
{
    const Grid &grid;
    const std::vector<int> &order, &sorted;
    std::vector<int> parent;

    std::vector<PersistencePair> pairs;
    std::vector<int> maxima;
    std::vector<SaddleInfo> saddles;

    BaselineSplitTree(const Grid &grid, const OrderField &o) : grid(grid), order(o.order), sorted(o.sorted), parent(grid.n)
    {
        iota(parent.begin(), parent.end(), 0);
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
        int n = grid.n;
        for (int r = n - 1; r >= 0; --r)
        {
            int v = sorted[r];
            int neighbors[NB];
            grid.neighbor(v, neighbors);
            int roots[NB], k = 0;

            for (int i = 0; i < NB; ++i)
            {
                int u = neighbors[i];
                if (u != -1 && order[u] > order[v])
                {
                    int root = find(u);
                    bool alreadyAdded = false;
                    for (int j = 0; j < k; ++j)
                    {
                        if (roots[j] == root)
                        {
                            alreadyAdded = true;
                            break;
                        }
                    }
                    if (!alreadyAdded)
                    {
                        roots[k++] = root;
                    }
                }
            }

            if (k == 0)
            {
                maxima.push_back(v);
            }
            else if (k == 1)
            {
                unite(v, roots[0]);
            }
            else
            {
                int root = roots[0];
                for (int i = 1; i < k; ++i)
                {
                    int otherRoot = roots[i];
                    if (order[otherRoot] > order[root])
                    {
                        root = otherRoot;
                    }
                }
                for (int i = 0; i < k; ++i)
                {
                    int otherRoot = roots[i];
                    if (otherRoot != root)
                    {
                        pairs.push_back(PersistencePair{otherRoot, v, root});
                        unite(v, otherRoot);
                    }
                }
                saddles.push_back(SaddleInfo{v, k - 1});
                unite(v, root);
            }
        }

        pairs.push_back(PersistencePair{sorted[n - 1], sorted[0], -1});
    }
};