#pragma once
#include <vector>

constexpr int NB = 14;
constexpr int OFFSETS[14][3] =
    {
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {1, 1, 0}, {-1, -1, 0}, {1, 0, 1}, {-1, 0, -1}, {0, 1, 1}, {0, -1, -1}, {1, 1, 1}, {-1, -1, -1}};

struct Grid
{
    int nx, ny, nz, n;
    int dv[NB];

    Grid(int nx, int ny, int nz) : nx(nx), ny(ny), nz(nz), n(nx * ny * nz)
    {
        for (int i = 0; i < NB; ++i)
        {
            dv[i] = OFFSETS[i][0] + nx * (OFFSETS[i][1] + ny * OFFSETS[i][2]);
        }
    }

    inline int id(int x, int y, int z) const
    {
        return x + nx * (y + ny * z);
    }

    inline void coords(int id, int &x, int &y, int &z) const
    {
        x = id % nx;
        y = (id / nx) % ny;
        z = id / (nx * ny);
    }

    inline bool isInterior(int x, int y, int z) const
    {
        return (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1);
    }

    inline bool inBounds(int x, int y, int z) const
    {
        return (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz);
    }

    inline int neighbor(int v, int out[NB]) const
    {
        int x, y, z;
        coords(v, x, y, z);
        if (isInterior(x, y, z))
        {
            for (int i = 0; i < NB; ++i)
            {
                out[i] = v + dv[i];
            }
            return NB;
        }
        else
        {
            int count = 0;
            for (int i = 0; i < NB; ++i)
            {
                if (inBounds(x + OFFSETS[i][0], y + OFFSETS[i][1], z + OFFSETS[i][2]))
                {
                    out[i] = v + dv[i];
                    ++count;
                }
                else
                {
                    out[i] = -1;
                }
            }
            return count;
        }
    }
};