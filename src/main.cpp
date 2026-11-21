#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <climits>
#include <fstream>
#include <memory>
#include <omp.h>
#include <tuple>

#include "io.h"
#include "grid.h"
#include "order.h"
#include "baseline_uf.h"
#include "path_compression.h"
#include "critical_points.h"
#include "extremum_graph.h"
#include "graph_split_tree.h"
#include "egp.h"
#include "output.h"

static void usage()
{
    std::fprintf(stderr,
                 "usage: extreem <file.raw> [--algo baseline] [--out <dir>]\n"
                 "  file name must look like  name_NXxNYxNZ_dtype.raw\n"
                 "  default --out is results/<file stem>/\n"
                 "  threads: set OMP_NUM_THREADS\n");
}

// Invariants that must hold for ANY input. Returns true if all pass.
static bool sanityCheckBaseline(const BaselineSplitTree &b, const OrderField &o, const std::vector<float> &f)
{
    const auto &order = o.order;
    const int n = (int)order.size();
    bool allOk = true;
    auto report = [&](const char *name, bool pass)
    {
        std::printf("  [%s] %s\n", pass ? "PASS" : "FAIL", name);
        allOk = allOk && pass;
    };

    std::vector<char> isMax(n, 0);
    for (int m : b.maxima)
        isMax[m] = 1;

    long long multSum = 0;
    for (const auto &s : b.saddles)
        multSum += s.multiplicity;

    std::vector<int> timesAsMax(n, 0);
    bool maxAboveSaddle = true, parentAboveMax = true, onlyRealMaxima = true;
    int rootCount = 0;
    bool rootIsGlobal = false;
    long long positive = 0;

    for (const auto &p : b.pairs)
    {
        if (!isMax[p.maxV])
            onlyRealMaxima = false;
        else
            timesAsMax[p.maxV]++;
        if (order[p.maxV] <= order[p.saddle])
            maxAboveSaddle = false;
        if (p.parentMax == -1)
        {
            rootCount++;
            rootIsGlobal = (p.maxV == o.sorted[n - 1] && p.saddle == o.sorted[0]);
        }
        else
        {
            if (!isMax[p.parentMax])
                onlyRealMaxima = false;
            if (order[p.parentMax] <= order[p.maxV])
                parentAboveMax = false;
        }
        if (f[p.maxV] - f[p.saddle] > 0.0f)
            positive++;
    }

    bool eachMaxOnce = true;
    for (int m : b.maxima)
        if (timesAsMax[m] != 1)
            eachMaxOnce = false;

    report("#pairs == #maxima", b.pairs.size() == b.maxima.size());
    report("sum(multiplicity) == #maxima - 1", multSum == (long long)b.maxima.size() - 1);
    report("every maximum appears in exactly one pair", eachMaxOnce);
    report("order[max] > order[saddle] for all pairs", maxAboveSaddle);
    report("order[parent] > order[max] for non-root pairs", parentAboveMax);
    report("exactly one root pair = (global max, global min)", rootCount == 1 && rootIsGlobal);
    report("pairs reference only real maxima", onlyRealMaxima);
    std::printf("  pairs with persistence > 0: %lld of %zu\n", positive, b.pairs.size());
    return allOk;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        usage();
        return 1;
    }

    std::string input = argv[1], algo = "baseline", outDir;
    for (int i = 2; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--algo" && i + 1 < argc)
            algo = argv[++i];
        else if (a == "--out" && i + 1 < argc)
            outDir = argv[++i];
        else
        {
            usage();
            return 1;
        }
    }

    try
    {
        const RawInfo info = parseFilename(input);
        if (outDir.empty())
            outDir = (algo == "vis" ? "vis/" : "results/") + info.stem;
        std::filesystem::create_directories(outDir);
        const std::string timingLog = "results/timings.csv";
        const int threads = omp_get_max_threads();

        std::printf("dataset %s  (%d x %d x %d, %s)  algo=%s  threads=%d\n",
                    info.stem.c_str(), info.nx, info.ny, info.nz, info.dtype.c_str(), algo.c_str(), threads);

        auto timed = [&](const std::string &step, auto &&fn)
        {
            double t0 = omp_get_wtime();
            fn();
            double dt = omp_get_wtime() - t0;
            std::printf("  %-12s %9.4f s\n", step.c_str(), dt);
            if (algo != "vis")
                appendTiming(timingLog, info.stem, algo, threads, step, dt);
        };

        std::vector<float> f;
        timed("load", [&]
              { f = loadRaw(input, info); });

        Grid grid(info.nx, info.ny, info.nz);

        OrderField o;
        timed("order", [&]
              { o = computeOrder(f); });

        if (algo == "baseline")
        {
            BaselineSplitTree b(grid, o);
            timed("baseline", [&]
                  { b.compute(); });

            std::printf("  maxima=%zu  saddles=%zu  pairs=%zu\n", b.maxima.size(), b.saddles.size(), b.pairs.size());
            bool ok = sanityCheckBaseline(b, o, f);

            writePairs(outDir + "/pairs_baseline.csv", b.pairs, o.order, f);
            writeMaxima(outDir + "/maxima_baseline.csv", b.maxima, o.order, f);
            writeSaddles(outDir + "/saddles_baseline.csv", b.saddles, o.order, f);
            std::printf("  wrote %s/{pairs,maxima,saddles}_baseline.csv\n", outDir.c_str());
            return ok ? 0 : 2;
        }
        else if (algo == "pc")
        {
            std::vector<int> dm;
            int rounds = 0;
            timed("pc", [&]
                  { dm = compressPath(o, grid, &rounds); });

            std::vector<int> maxima;
            for (int v = 0; v < grid.n; ++v)
                if (dm[v] == v)
                    maxima.push_back(v);

            std::printf("  maxima=%zu  rounds=%d\n", maxima.size(), rounds);
            writeMaxima(outDir + "/maxima_pc.csv", maxima, o.order, f);
            writeBinaryInt(outDir + "/dm.bin", dm);
            return 0;
        }
        else if (algo == "cp")
        {
            std::vector<int> dm;
            int rounds = 0;
            timed("pc", [&]
                  { dm = compressPath(o, grid, &rounds); });

            static LinkTables lt;
            timed("lut_build", [&]
                  { buildLUT(lt); });

            CriticalPoints cpS, cpL;
            timed("cp_simple", [&]
                  { cpS = classifyCriticalPoints(grid, o, dm, lt, false); });
            timed("cp_lut", [&]
                  { cpL = classifyCriticalPoints(grid, o, dm, lt, true); });

            auto byV = [](const SaddleInfo &a, const SaddleInfo &b)
            { return a.v < b.v; };
            std::sort(cpS.saddles.begin(), cpS.saddles.end(), byV);
            std::sort(cpL.saddles.begin(), cpL.saddles.end(), byV);
            std::sort(cpS.maxima.begin(), cpS.maxima.end());
            std::sort(cpL.maxima.begin(), cpL.maxima.end());

            bool same = cpS.maxima == cpL.maxima && cpS.saddles.size() == cpL.saddles.size() &&
                        std::equal(cpS.saddles.begin(), cpS.saddles.end(), cpL.saddles.begin(),
                                   [](const SaddleInfo &a, const SaddleInfo &b)
                                   { return a.v == b.v && a.multiplicity == b.multiplicity; });

            // maxima from classification must be exactly the vertices with dm[v] == v
            std::vector<int> pcMax;
            for (int v = 0; v < grid.n; ++v)
                if (dm[v] == v)
                    pcMax.push_back(v);
            bool maxConsistent = (pcMax == cpL.maxima);

            long long mult = 0;
            for (auto &s : cpL.saddles)
                mult += s.multiplicity;

            std::printf("  maxima=%zu  saddles=%zu  total_multiplicity=%lld  pc_rounds=%d\n",
                        cpL.maxima.size(), cpL.saddles.size(), mult, rounds);
            std::printf("  [%s] simple == LUT\n", same ? "PASS" : "FAIL");
            std::printf("  [%s] maxima == {v : dm[v] == v}\n", maxConsistent ? "PASS" : "FAIL");

            writeMaxima(outDir + "/maxima_cp.csv", cpL.maxima, o.order, f);
            writeSaddles(outDir + "/saddles_cp.csv", cpL.saddles, o.order, f);
            return (same && maxConsistent) ? 0 : 2;
        }
        else if (algo == "graph")
        {
            std::vector<int> dm;
            timed("pc", [&]
                  { dm = compressPath(o, grid); });

            static LinkTables lt;
            buildLUT(lt);
            CriticalPoints cp;
            timed("cp_lut", [&]
                  { cp = classifyCriticalPoints(grid, o, dm, lt, true); });

            ExtremumGraph G;
            timed("graph", [&]
                  { G = buildGraph(grid, o, dm, cp); });

            GraphSplitTree gst(G, o);
            timed("graph_uf", [&]
                  { gst.compute(); });

            const long long VG = (long long)G.maxima.size() + (long long)G.saddles.size();
            const long long EG = (long long)G.nbr.size();
            const long long VK = grid.n, EK = gridEdgeCount(grid);
            const double ratio = 100.0 * (double)(VG + EG) / (double)(VK + EK);

            int minDeg = INT_MAX, maxDeg = 0;
            for (size_t i = 0; i < G.saddles.size(); ++i)
            {
                int d = G.offset[i + 1] - G.offset[i];
                minDeg = std::min(minDeg, d);
                maxDeg = std::max(maxDeg, d);
            }
            if (G.saddles.empty())
                minDeg = 2;

            std::printf("  |V(G)|=%lld  |E(G)|=%lld   |V(K)|=%lld  |E(K)|=%lld   ratio=%.4f%%\n", VG, EG, VK, EK, ratio);
            std::printf("  saddle degree: min=%d max=%d   pairs=%zu\n", minDeg, maxDeg, gst.pairs.size());
            std::printf("  [%s] every saddle has >= 2 edges\n", minDeg >= 2 ? "PASS" : "FAIL");

            writePairs(outDir + "/pairs_graph.csv", gst.pairs, o.order, f);

            const std::string statsPath = "results/graph_stats.csv";
            const bool isNew = !std::filesystem::exists(statsPath);
            std::ofstream st(statsPath, std::ios::app);
            if (isNew)
                st << "dataset,maxima,saddles,VG,EG,VK,EK,ratio_percent\n";
            st << info.stem << ',' << G.maxima.size() << ',' << G.saddles.size() << ','
               << VG << ',' << EG << ',' << VK << ',' << EK << ',' << ratio << '\n';

            return minDeg >= 2 ? 0 : 2;
        }
        else if (algo == "egp")
        {
            std::vector<int> dm;
            timed("pc", [&]
                  { dm = compressPath(o, grid); });

            static LinkTables lt;
            buildLUT(lt);
            CriticalPoints cp;
            timed("cp_lut", [&]
                  { cp = classifyCriticalPoints(grid, o, dm, lt, true); });

            ExtremumGraph G;
            timed("graph", [&]
                  { G = buildGraph(grid, o, dm, cp); });

            std::unique_ptr<EGP> egpS, egpP;
            timed("egp_setup", [&]
                  { egpS = std::make_unique<EGP>(G, o); });
            timed("egp_serial", [&]
                  { egpS->egpSerial(); });
            egpP = std::make_unique<EGP>(G, o);
            timed("egp_par", [&]
                  { egpP->egpParallel(); });

            auto canon = [](std::vector<PersistencePair> p)
            {
                std::sort(p.begin(), p.end(), [](const PersistencePair &a, const PersistencePair &b)
                          { return std::tie(a.maxV, a.saddle, a.parentMax) < std::tie(b.maxV, b.saddle, b.parentMax); });
                return p;
            };
            auto a = canon(egpS->pairs), b = canon(egpP->pairs);
            bool same = a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
                                                           [](const PersistencePair &x, const PersistencePair &y)
                                                           { return x.maxV == y.maxV && x.saddle == y.saddle && x.parentMax == y.parentMax; });

            std::printf("  egp rounds: serial=%d parallel=%d  pairs=%zu\n", egpS->rounds, egpP->rounds, egpP->pairs.size());
            std::printf("  [%s] parallel == serial\n", (same && egpS->rounds == egpP->rounds) ? "PASS" : "FAIL");
            writePairs(outDir + "/pairs_egp.csv", egpP->pairs, o.order, f);
            return same ? 0 : 2;
        }

        else if (algo == "vis")
        {
            // Writes everything python/export_vtk.py needs. Not timed, no benchmark logs.
            std::vector<int> dm = compressPath(o, grid);
            static LinkTables lt;
            buildLUT(lt);
            CriticalPoints cp = classifyCriticalPoints(grid, o, dm, lt, true);
            ExtremumGraph G = buildGraph(grid, o, dm, cp);
            EGP egp(G, o);
            egp.egpParallel();

            writeBinaryInt(outDir + "/dm.bin", dm);
            writePairs(outDir + "/pairs_egp.csv", egp.pairs, o.order, f);
            writeMaxima(outDir + "/maxima.csv", cp.maxima, o.order, f);
            writeSaddles(outDir + "/saddles.csv", cp.saddles, o.order, f);

            std::ofstream ge(outDir + "/graph_edges.csv");
            ge << "saddle_id,max_id\n";
            for (size_t i = 0; i < G.saddles.size(); ++i)
                for (int e = G.offset[i]; e < G.offset[i + 1]; ++e)
                    ge << G.saddles[i] << ',' << G.nbr[e] << '\n';

            std::printf("  wrote vis inputs to %s\n", outDir.c_str());
            return 0;
        }

        std::fprintf(stderr, "unknown --algo '%s'\n", algo.c_str());
        return 1;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}