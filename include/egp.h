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
    const ExtremumGraph &eg;
    const std::vector<int> &order, &sorted;

    int M = 0, S = 0;
    std::vector<int> maxV;
    std::vector<int> idx;

    std::vector<PersistencePair> pairs;
    int rounds = 0;

    EGP(const ExtremumGraph &g, const OrderField &o) : eg(g), order(o.order), sorted(o.sorted)
    {
        M = (int)eg.maxima.size();
        S = (int)eg.saddles.size();
        maxV = eg.maxima;
        std::sort(maxV.begin(), maxV.end(), [&](int a, int b)
                  { return order[a] > order[b]; });
        idx.assign(order.size(), -1);
        for (int m = 0; m < M; ++m)
            idx[maxV[m]] = m;
    }

    void egpSerial()
    {
        std::vector<int> nbr(eg.nbr.size());
        for (size_t e = 0; e < eg.nbr.size(); ++e)
            nbr[e] = idx[eg.nbr[e]];
        std::vector<int> len(S);
        for (int s = 0; s < S; ++s)
            len[s] = eg.offset[s + 1] - eg.offset[s];

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
                const int b = eg.offset[s], e = b + len[s];
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
                const int b = eg.offset[s], e = b + len[s];
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
                const int b = eg.offset[s];
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
            pairs.push_back({maxV[m], eg.saddles[s], maxV[x]});
        }

        for (int m = 1; m < M; ++m)
            if (!dead[m])
                throw std::runtime_error("EGP: a non-global maximum survived (bug)");
        if (M == 0 || maxV[0] != sorted.back())
            throw std::runtime_error("EGP: survivor is not the global maximum (bug)");
        pairs.push_back({maxV[0], sorted[0], -1});
    }

    static inline void atomicMin(int *p, int v)
    {
        int cur = __atomic_load_n(p, __ATOMIC_RELAXED);
        while (v < cur &&
               !__atomic_compare_exchange_n(p, &cur, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
        {
        }
    }

    void egpParallel()
    {
        const long long E = (long long)eg.nbr.size();
        std::vector<int> nbr(E), len(S);
#pragma omp parallel for schedule(static)
        for (long long e = 0; e < E; ++e)
            nbr[e] = idx[eg.nbr[e]];
#pragma omp parallel for schedule(static)
        for (int s = 0; s < S; ++s)
            len[s] = eg.offset[s + 1] - eg.offset[s];

        std::vector<int> R(M), L(M), deathSaddle(M, -1), target(M, -1);
        std::vector<char> dead(M, 0);
#pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m)
            R[m] = m;

        std::vector<int> active(S), nextActive;
#pragma omp parallel for schedule(static)
        for (int s = 0; s < S; ++s)
            active[s] = s;

        int alive = M;
        rounds = 0;

        while (alive > 1)
        {
            ++rounds;
            const int A = (int)active.size();

#pragma omp parallel for schedule(static)
            for (int m = 0; m < M; ++m)
                L[m] = INT_MAX;
#pragma omp parallel for schedule(dynamic, 1024)
            for (int a = 0; a < A; ++a)
            {
                const int s = active[a];
                const int b = eg.offset[s], e = b + len[s];
                for (int k = b; k < e; ++k)
                    atomicMin(&L[nbr[k]], s);
            }

            int paired = 0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : paired)
            for (int m = 0; m < M; ++m)
            {
                if (dead[m] || L[m] == INT_MAX)
                    continue;
                const int s = L[m];
                const int b = eg.offset[s], e = b + len[s];
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
                    ++paired;
                }
            }
            if (paired == 0)
                throw std::runtime_error("EGP parallel: a round formed no pair (bug)");
            alive -= paired;
\
#pragma omp parallel for schedule(dynamic, 1024)
            for (int m = 0; m < M; ++m)
            {
                if (!dead[m])
                    continue;
                int x;
#pragma omp atomic read
                x = R[m];
                while (dead[x])
                {
                    int y;
#pragma omp atomic read
                    y = R[x];
                    x = y;
                }
#pragma omp atomic write
                R[m] = x;
            }

            nextActive.clear();
#pragma omp parallel
            {
                std::vector<int> local;
#pragma omp for schedule(dynamic, 1024) nowait
                for (int a = 0; a < A; ++a)
                {
                    const int s = active[a];
                    const int b = eg.offset[s];
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
                        local.push_back(s);
                }
#pragma omp critical
                nextActive.insert(nextActive.end(), local.begin(), local.end());
            }
            active.swap(nextActive);
        }

        std::vector<int> parent(M, -1);
#pragma omp parallel for schedule(dynamic, 1024)
        for (int m = 0; m < M; ++m)
        {
            if (!dead[m])
                continue;
            const int s = deathSaddle[m];
            int x = target[m];
            while (dead[x] && deathSaddle[x] <= s)
                x = target[x];
            parent[m] = x;
        }

        for (int m = 1; m < M; ++m)
            if (!dead[m])
                throw std::runtime_error("EGP parallel: a non-global maximum survived (bug)");
        if (M == 0 || maxV[0] != sorted.back())
            throw std::runtime_error("EGP parallel: survivor is not the global maximum (bug)");

        pairs.clear();
        pairs.reserve(M);
        for (int m = 1; m < M; ++m)
            pairs.push_back({maxV[m], eg.saddles[deathSaddle[m]], maxV[parent[m]]});
        pairs.push_back({maxV[0], sorted[0], -1});
    }
};