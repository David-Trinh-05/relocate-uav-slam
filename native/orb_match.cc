// Exact mutual nearest-neighbor matching for 32-byte ORB descriptors.

#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int relocate_match_orb32(const std::uint8_t* left, int left_count,
                                    const std::uint8_t* right, int right_count, int max_distance, int* output)
{
    const int beyond_limit = max_distance + 1;
    std::vector<int> right_distances(right_count, beyond_limit), right_partners(right_count, -1);
    std::vector<int> left_distances(left_count, beyond_limit), left_partners(left_count, -1);
    for (int i = 0; i < left_count; ++i)
    {
        std::uint64_t a[4];
        std::memcpy(a, left + 32 * i, sizeof(a));
        int best_distance = beyond_limit, best_partner = -1;
        for (int j = 0; j < right_count; ++j)
        {
            std::uint64_t b[4];
            std::memcpy(b, right + 32 * j, sizeof(b));
            int distance = __builtin_popcountll(a[0] ^ b[0]) + __builtin_popcountll(a[1] ^ b[1]);
            if (distance > max_distance)
                continue;
            distance += __builtin_popcountll(a[2] ^ b[2]) + __builtin_popcountll(a[3] ^ b[3]);
            // Strict comparison gives the first descriptor priority on a tie.
            if (distance < best_distance)
            {
                best_distance = distance;
                best_partner = j;
            }
            if (distance < right_distances[j])
            {
                right_distances[j] = distance;
                right_partners[j] = i;
            }
        }
        left_distances[i] = best_distance;
        left_partners[i] = best_partner;
    }

    int count = 0;
    for (int i = 0; i < left_count; ++i)
    {
        const int j = left_partners[i];
        if (j < 0 || right_partners[j] != i)
            continue;
        output[3 * count] = i;
        output[3 * count + 1] = j;
        output[3 * count + 2] = left_distances[i];
        ++count;
    }
    return count;
}
