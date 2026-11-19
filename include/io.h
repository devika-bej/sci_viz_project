#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Parsed from a name like  bonsai_256x256x256_uint8.raw
struct RawInfo
{
    std::string stem;  // bonsai_256x256x256_uint8   (used for output folder names)
    std::string name;  // bonsai
    int nx = 0, ny = 0, nz = 0;
    std::string dtype; // uint8 | uint16 | float32 | float64
};

inline size_t bytesPerValue(const std::string &dtype)
{
    if (dtype == "uint8")   return 1;
    if (dtype == "uint16")  return 2;
    if (dtype == "float32") return 4;
    if (dtype == "float64") return 8;
    throw std::runtime_error("unsupported dtype '" + dtype + "'");
}

inline RawInfo parseFilename(const std::string &path)
{
    RawInfo info;
    info.stem = std::filesystem::path(path).stem().string();
    const std::string &s = info.stem;

    size_t p2 = s.rfind('_');                         // before dtype
    if (p2 == std::string::npos || p2 == 0)
        throw std::runtime_error("filename must look like name_NXxNYxNZ_dtype.raw: " + s);
    size_t p1 = s.rfind('_', p2 - 1);                 // before dims
    if (p1 == std::string::npos)
        throw std::runtime_error("filename must look like name_NXxNYxNZ_dtype.raw: " + s);

    info.name  = s.substr(0, p1);
    info.dtype = s.substr(p2 + 1);
    std::string dims = s.substr(p1 + 1, p2 - p1 - 1);
    if (std::sscanf(dims.c_str(), "%dx%dx%d", &info.nx, &info.ny, &info.nz) != 3 ||
        info.nx <= 0 || info.ny <= 0 || info.nz <= 0)
        throw std::runtime_error("cannot parse dimensions '" + dims + "'");

    bytesPerValue(info.dtype);                        // validates dtype early
    return info;
}

// Loads the raw volume (x fastest, little-endian) and converts every value to float.
inline std::vector<float> loadRaw(const std::string &path, const RawInfo &info)
{
    const size_t n   = (size_t)info.nx * info.ny * info.nz;
    const size_t bpv = bytesPerValue(info.dtype);

    const auto fileSize = std::filesystem::file_size(path);
    if (fileSize != n * bpv)
        throw std::runtime_error("size mismatch: file has " + std::to_string(fileSize) +
                                 " bytes, expected " + std::to_string(n * bpv));

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<char> buf(fileSize);
    in.read(buf.data(), (std::streamsize)fileSize);
    if (!in) throw std::runtime_error("read failed: " + path);

    std::vector<float> f(n);
    if (info.dtype == "uint8")
    {
        for (size_t i = 0; i < n; ++i) f[i] = (float)(unsigned char)buf[i];
    }
    else if (info.dtype == "uint16")
    {
        std::vector<uint16_t> tmp(n);
        std::memcpy(tmp.data(), buf.data(), n * 2);
        for (size_t i = 0; i < n; ++i) f[i] = (float)tmp[i];
    }
    else if (info.dtype == "float32")
    {
        std::memcpy(f.data(), buf.data(), n * 4);
    }
    else // float64
    {
        std::vector<double> tmp(n);
        std::memcpy(tmp.data(), buf.data(), n * 8);
        for (size_t i = 0; i < n; ++i) f[i] = (float)tmp[i];
    }
    return f;
}