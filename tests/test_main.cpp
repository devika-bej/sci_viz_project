#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>
#include "grid.h"
#include "order.h"
#include "baseline_uf.h"
#include "path_compression.h"
#include "critical_points.h"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

// ======================= Phase 1: baseline =======================

static bool hasPair(const BaselineSplitTree &b, int m, int s, int p)
{
    for (auto &q : b.pairs) if (q.maxV == m && q.saddle == s && q.parentMax == p) return true;
    return false;
}

static void test1D()
{
    std::vector<float> f = {2, 9, 3, 7, 1};
    Grid g(5, 1, 1); OrderField o = computeOrder(f);
    BaselineSplitTree b(g, o); b.compute();
    CHECK(b.maxima.size() == 2);
    CHECK(b.saddles.size() == 1 && b.saddles[0].v == 2 && b.saddles[0].multiplicity == 1);
    CHECK(b.pairs.size() == 2);
    CHECK(hasPair(b, 3, 2, 1));
    CHECK(hasPair(b, 1, 4, -1));
}

static void testDiagonal()
{
    {
        std::vector<float> f = {9, 0, 0, 8};
        Grid g(2, 2, 1); OrderField o = computeOrder(f);
        BaselineSplitTree b(g, o); b.compute();
        CHECK(b.maxima.size() == 1);
        CHECK(b.pairs.size() == 1 && hasPair(b, 0, 1, -1));
    }
    {
        std::vector<float> f = {0, 9, 8, 0};
        Grid g(2, 2, 1); OrderField o = computeOrder(f);
        BaselineSplitTree b(g, o); b.compute();
        CHECK(b.maxima.size() == 2);
        CHECK(hasPair(b, 2, 3, 1));
        CHECK(hasPair(b, 1, 0, -1));
    }
}

static void testMultiSaddle()
{
    std::vector<float> f = {1, 9, 2, 8, 5, 0, 3, 4, 7};
    Grid g(3, 3, 1); OrderField o = computeOrder(f);
    BaselineSplitTree b(g, o); b.compute();
    CHECK(b.maxima.size() == 3);
    CHECK(b.saddles.size() == 1 && b.saddles[0].v == 4 && b.saddles[0].multiplicity == 2);
    CHECK(hasPair(b, 3, 4, 1));
    CHECK(hasPair(b, 8, 4, 1));
    CHECK(hasPair(b, 1, 5, -1));
}

// ======================= Phase 1: path compression =======================

static void testPathCompression()
{
    std::vector<float> f = {2, 9, 3, 7, 1};
    Grid g(5, 1, 1); OrderField o = computeOrder(f);
    std::vector<int> dm = compressPath(o, g);
    CHECK((dm == std::vector<int>{1, 1, 1, 3, 3}));
}

// ======================= Phase 2: link tables =======================

static void testLinkTables(const LinkTables &lt)
{
    int total = 0;
    bool symmetric = true;
    for (int i = 0; i < NB; ++i)
    {
        total += __builtin_popcount(lt.adjMask[i]);
        for (int j = 0; j < NB; ++j)
            if (((lt.adjMask[i] >> j) & 1) != ((lt.adjMask[j] >> i) & 1)) symmetric = false;
    }
    CHECK(total == 72);
    CHECK(symmetric);
    CHECK(lt.lut[0] == 0);
    for (int i = 0; i < NB; ++i) CHECK(lt.lut[1 << i] == 1);
    CHECK(lt.lut[(1 << NB) - 1] == 1);
    CHECK(lt.lut[(1 << 0) | (1 << 1)] == 2);   // +x, -x: not adjacent
    CHECK(lt.lut[(1 << 0) | (1 << 6)] == 1);   // +x, (1,1,0): adjacent
}

// ======================= Phase 2: classification =======================

static void checkCP(const char *name, const std::vector<float> &f, int nx, int ny, int nz,
                    std::vector<int> expMax, std::vector<std::pair<int, int>> expSad,
                    const LinkTables &lt)
{
    Grid g(nx, ny, nz);
    OrderField o = computeOrder(f);
    std::vector<int> dm = compressPath(o, g);
    std::sort(expMax.begin(), expMax.end());
    std::sort(expSad.begin(), expSad.end());

    for (bool useLUT : {false, true})
    {
        CriticalPoints cp = classifyCriticalPoints(g, o, dm, lt, useLUT);
        std::sort(cp.maxima.begin(), cp.maxima.end());
        std::vector<std::pair<int, int>> sad;
        for (auto &s : cp.saddles) sad.push_back({s.v, s.multiplicity});
        std::sort(sad.begin(), sad.end());

        if (cp.maxima != expMax) { std::printf("FAIL %s (useLUT=%d): maxima\n", name, useLUT); ++failures; }
        if (sad != expSad)       { std::printf("FAIL %s (useLUT=%d): saddles\n", name, useLUT); ++failures; }
    }
}

static void testClassification(const LinkTables &lt)
{
    checkCP("1D",            {2, 9, 3, 7, 1},             5, 1, 1, {1, 3},    {{2, 1}}, lt);
    checkCP("2x2 diag-edge", {9, 0, 0, 8},                2, 2, 1, {0},       {},       lt);
    checkCP("2x2 no-edge",   {0, 9, 8, 0},                2, 2, 1, {1, 2},    {{3, 1}}, lt);
    checkCP("3x3 multi",     {1, 9, 2, 8, 5, 0, 3, 4, 7}, 3, 3, 1, {1, 3, 8}, {{4, 2}}, lt);
}

int main()
{
    test1D();
    testDiagonal();
    testMultiSaddle();
    testPathCompression();

    static LinkTables lt;   // ~16 KB, static to keep it off the stack
    buildLUT(lt);
    testLinkTables(lt);
    testClassification(lt);

    std::printf(failures ? "%d FAILURES\n" : "ALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}