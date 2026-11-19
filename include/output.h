#pragma once
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "baseline_uf.h"

inline std::ofstream openOut(const std::string &path)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.precision(9); // enough digits to round-trip a float exactly
    return out;
}

// Rows sorted by (saddle order desc, max order desc) -> deterministic, diff-able files.
// The root pair (saddle = global min) naturally ends up last.
inline void writePairs(const std::string &path, std::vector<PersistencePair> pairs,
                       const std::vector<int> &order, const std::vector<float> &f)
{
    std::sort(pairs.begin(), pairs.end(), [&](const PersistencePair &a, const PersistencePair &b) {
        if (order[a.saddle] != order[b.saddle]) return order[a.saddle] > order[b.saddle];
        return order[a.maxV] > order[b.maxV];
    });
    auto out = openOut(path);
    out << "max_id,saddle_id,parent_max_id,max_order,saddle_order,max_value,saddle_value,persistence\n";
    for (const auto &p : pairs)
        out << p.maxV << ',' << p.saddle << ',' << p.parentMax << ','
            << order[p.maxV] << ',' << order[p.saddle] << ','
            << f[p.maxV] << ',' << f[p.saddle] << ','
            << (double)f[p.maxV] - (double)f[p.saddle] << '\n';
}

inline void writeMaxima(const std::string &path, std::vector<int> maxima,
                        const std::vector<int> &order, const std::vector<float> &f)
{
    std::sort(maxima.begin(), maxima.end(), [&](int a, int b) { return order[a] > order[b]; });
    auto out = openOut(path);
    out << "max_id,max_order,max_value\n";
    for (int m : maxima) out << m << ',' << order[m] << ',' << f[m] << '\n';
}

inline void writeSaddles(const std::string &path, std::vector<SaddleInfo> saddles,
                         const std::vector<int> &order, const std::vector<float> &f)
{
    std::sort(saddles.begin(), saddles.end(),
              [&](const SaddleInfo &a, const SaddleInfo &b) { return order[a.v] > order[b.v]; });
    auto out = openOut(path);
    out << "saddle_id,multiplicity,saddle_order,saddle_value\n";
    for (const auto &s : saddles) out << s.v << ',' << s.multiplicity << ',' << order[s.v] << ',' << f[s.v] << '\n';
}

// Raw int32 array (same x-fastest layout as the input). Used for dm.bin / segmentations.
inline void writeBinaryInt(const std::string &path, const std::vector<int> &a)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write(reinterpret_cast<const char *>(a.data()), (std::streamsize)(a.size() * sizeof(int)));
}

// Appends one row to a shared timings log; writes the header if the file is new.
inline void appendTiming(const std::string &path, const std::string &dataset, const std::string &algo,
                         int threads, const std::string &step, double seconds)
{
    bool isNew = !std::filesystem::exists(path);
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot write " + path);
    if (isNew) out << "dataset,algo,threads,step,seconds\n";
    out << dataset << ',' << algo << ',' << threads << ',' << step << ',' << seconds << '\n';
}