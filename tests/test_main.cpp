#include <cstdio>
#include <vector>
#include "grid.h"
#include "order.h"
#include "baseline_uf.h"

static int failures = 0;
#define CHECK(cond)                                                     \
    do                                                                  \
    {                                                                   \
        if (!(cond))                                                    \
        {                                                               \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

static bool hasPair(const BaselineSplitTree &b, int m, int s, int p)
{
    for (auto &q : b.pairs)
        if (q.maxV == m && q.saddle == s && q.parentMax == p)
            return true;
    return false;
}

// 1D:  f = 2 9 3 7 1   -> peak 7 (id3) dies at 3 (id2) into peak 9 (id1)
static void test1D()
{
    std::vector<float> f = {2, 9, 3, 7, 1};
    Grid g(5, 1, 1);
    OrderField o = computeOrder(f);
    BaselineSplitTree b(g, o);
    b.compute();
    CHECK(b.maxima.size() == 2);
    CHECK(b.saddles.size() == 1 && b.saddles[0].v == 2 && b.saddles[0].multiplicity == 1);
    CHECK(b.pairs.size() == 2);
    CHECK(hasPair(b, 3, 2, 1));
    CHECK(hasPair(b, 1, 4, -1));
}

// 2D 2x2:  ids 0 1 / 2 3.   (1,1)-(0,0) IS a Freudenthal edge, (1,0)-(0,1) is NOT.
static void testDiagonal()
{
    { // 9 at id0, 8 at id3: connected by the diagonal -> only one maximum
        std::vector<float> f = {9, 0, 0, 8};
        Grid g(2, 2, 1);
        OrderField o = computeOrder(f);
        BaselineSplitTree b(g, o);
        b.compute();
        CHECK(b.maxima.size() == 1);
        CHECK(b.pairs.size() == 1 && hasPair(b, 0, 1, -1)); // min = id1 (tie with id2, lower id is lower)
    }
    { // 9 at id1, 8 at id2: NOT neighbours -> two maxima, merge at id3
        std::vector<float> f = {0, 9, 8, 0};
        Grid g(2, 2, 1);
        OrderField o = computeOrder(f);
        BaselineSplitTree b(g, o);
        b.compute();
        CHECK(b.maxima.size() == 2);
        CHECK(hasPair(b, 2, 3, 1)); // id3 beats id0 on the tie
        CHECK(hasPair(b, 1, 0, -1));
    }
}

// 3x3 multi-saddle: peaks 9 (id1), 8 (id3), 7 (id8) all meet at centre id4 (value 5)
//   ids:  0 1 2      f:  1 9 2
//         3 4 5          8 5 0
//         6 7 8          3 4 7
static void testMultiSaddle()
{
    std::vector<float> f = {1, 9, 2, 8, 5, 0, 3, 4, 7};
    Grid g(3, 3, 1);
    OrderField o = computeOrder(f);
    BaselineSplitTree b(g, o);
    b.compute();
    CHECK(b.maxima.size() == 3);
    CHECK(b.saddles.size() == 1 && b.saddles[0].v == 4 && b.saddles[0].multiplicity == 2);
    CHECK(hasPair(b, 3, 4, 1));
    CHECK(hasPair(b, 8, 4, 1));
    CHECK(hasPair(b, 1, 5, -1));
}

int main()
{
    test1D();
    testDiagonal();
    testMultiSaddle();
    std::printf(failures ? "%d FAILURES\n" : "ALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}