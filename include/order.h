#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <parallel/algorithm>

struct OrderField
{
    std::vector<int> order;
    std::vector<int> sorted;
};

inline OrderField computeOrder(const std::vector<float> &f)
{
    OrderField result;
    int n = f.size();
    result.order.resize(n);
    result.sorted.resize(n);

    std::iota(result.sorted.begin(), result.sorted.end(), 0);
    __gnu_parallel::sort(result.sorted.begin(), result.sorted.end(), [&](int i, int j)
                         { return f[i] < f[j] || (f[i] == f[j] && i < j); });

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        result.order[result.sorted[i]] = i;
    }

    return result;
}

inline bool higher(const OrderField &o, int i, int j)
{
    return o.order[i] > o.order[j];
}

inline bool isValidOrder(const OrderField &o)
{
    if (o.order.size() != o.sorted.size())
        return false;

    const int n = static_cast<int>(o.order.size());
    std::vector<bool> rankUsed(n, false);
    for (int v = 0; v < n; ++v)
    {
        const int rank = o.order[v];
        if (rank < 0 || rank >= n || rankUsed[rank])
            return false;
        rankUsed[rank] = true;
        if (o.sorted[rank] != v)
            return false;
    }

    for (int rank = 0; rank < n; ++rank)
    {
        const int v = o.sorted[rank];
        if (v < 0 || v >= n || o.order[v] != rank)
            return false;
    }

    return true;
}

inline bool isValidOrder(const OrderField &o, const std::vector<float> &f)
{
    if (!isValidOrder(o) || f.size() != o.sorted.size())
        return false;

    for (size_t r = 0; r + 1 < o.sorted.size(); ++r)
    {
        const int current = o.sorted[r];
        const int next = o.sorted[r + 1];
        if (!(f[current] < f[next] || (f[current] == f[next] && current < next)))
            return false;
    }

    return true;
}