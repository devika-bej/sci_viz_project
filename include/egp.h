#pragma once
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>
#include <vector>
#include "order.h"
#include "extremum_graph.h"

struct EGP
{
    const ExtremumGraph &G;
    const std::vector<int> &order, &sorted;

    int M = 0, S = 0;
    std::vector<int> maxV;
    std::vector<int> idx;

    std::vector<PersistencePair> pairs;
    int rounds = 0;

    EGP(const ExtremumGraph &g, const OrderField &o) : G(g), order(o.order), sorted(o.sorted)
    {
        M = (int)G.maxima.size();
        S = (int)G.saddles.size();
        maxV = G.maxima;
        std::sort(maxV.begin(), maxV.end(), [&](int a, int b)
                  { return order[a] > order[b]; });
        idx.assign(order.size(), -1);
        for (int m = 0; m < M; ++m)
            idx[maxV[m]] = m;
    }

    void egpSerial()
    {
        std::vector<int> nbr(G.nbr.size());
        for (size_t e = 0; e < G.nbr.size(); ++e)
            nbr[e] = idx[G.nbr[e]];
        std::vector<int> len(S);
        for (int s = 0; s < S; ++s)
            len[s] = G.offset[s + 1] - G.offset[s];

        std::vector<int> R(M), L(M), deathSaddle(M, -1), target(M, -1);
        std::vector<char> dead(M, 0);
        for (int m = 0; m < M; ++m)
            R[m] = m;

        std::vector<int> active(S), nextActive;
        for (int s = 0; s < S; ++s)
            active[s] = s;

        std::vector<std::pair<int, int>> branches;
        branches.reserve(M);
        int alive = M;
        rounds = 0;

        while (alive > 1)
        {
            ++rounds;
            std::fill(L.begin(), L.end(), INT_MAX);
            for (int s : active)
            {
                const int b = G.offset[s], e = b + len[s];
                for (int k = b; k < e; ++k)
                    if (s < L[nbr[k]])
                        L[nbr[k]] = s;
            }

            int paired = 0;
            for (int m = 0; m < M; ++m)
            {
                if (dead[m] || L[m] == INT_MAX)
                    continue;
                const int s = L[m];
                const int b = G.offset[s], e = b + len[s];
                int lo = nbr[b], hi = nbr[b];
                for (int k = b + 1; k < e; ++k)
                {
                    lo = std::min(lo, nbr[k]);
                    hi = std::max(hi, nbr[k]);
                }
                if (m == hi)
                {
                    dead[m] = 1;
                    deathSaddle[m] = s;
                    target[m] = lo;
                    R[m] = lo;
                    branches.push_back({m, s});
                    --alive;
                    ++paired;
                }
            }
            if (paired == 0)
                throw std::runtime_error("EGP: a round formed no pair (bug)");

            for (int m = 0; m < M; ++m)
                R[m] = R[R[m]];

            nextActive.clear();
            for (int s : active)
            {
                const int b = G.offset[s];
                int c = 0;
                for (int k = b; k < b + len[s]; ++k)
                {
                    const int r = R[nbr[k]];
                    bool seen = false;
                    for (int j = b; j < b + c; ++j)
                        if (nbr[j] == r)
                        {
                            seen = true;
                            break;
                        }
                    if (!seen)
                        nbr[b + c++] = r;
                }
                len[s] = c;
                if (c >= 2)
                    nextActive.push_back(s);
            }
            active.swap(nextActive);
        }

        pairs.clear();
        pairs.reserve(branches.size() + 1);
        for (const auto &[m, s] : branches)
        {
            int x = target[m];
            while (dead[x] && deathSaddle[x] <= s)
                x = target[x];
            pairs.push_back({maxV[m], G.saddles[s], maxV[x]});
        }

        for (int m = 1; m < M; ++m)
            if (!dead[m])
                throw std::runtime_error("EGP: a non-global maximum survived (bug)");
        if (M == 0 || maxV[0] != sorted.back())
            throw std::runtime_error("EGP: survivor is not the global maximum (bug)");
        pairs.push_back({maxV[0], sorted[0], -1});
    }
};