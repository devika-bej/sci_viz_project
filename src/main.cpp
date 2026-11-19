#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include <omp.h>

#include "io.h"
#include "grid.h"
#include "order.h"
#include "baseline_uf.h"
#include "path_compression.h"
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
            outDir = "results/" + info.stem;
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

        std::fprintf(stderr, "unknown --algo '%s'\n", algo.c_str());
        return 1;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}