#pragma once

#include "engine.h"
#include <vector>

namespace eqt {
// Negative strengths scale the projection that removes the direction from every block (-1 removes it); positive
// strengths scale the mean difference added at its block. Larger additions left the 35B model incoherent.
constexpr double steering_minimum = -1.0;
constexpr double steering_maximum = 2.0;

struct SteeringVector {
    Json metadata;
    int layer;  // block that receives positive strengths
    int embedding;
    std::vector<int> blocks;   // block of each direction, ascending: {layer}, or 1..layer_count
    std::vector<float> values; // one row of `embedding` floats per block

    const float *row(int block) const;
};

// Parses bounded, immutable bytes and verifies their fingerprint before interpreting GGUF metadata.
SteeringVector read_steering(const std::string &path, const Json &manifest);
} // namespace eqt
