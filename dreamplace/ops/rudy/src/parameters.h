#ifndef DREAMPLACE_ROUTABILITY_PARAMETERS_H
#define DREAMPLACE_ROUTABILITY_PARAMETERS_H

/// The function returns the mean wirelength weight for a net with 'num_pins' pins using wiring distribution map (WDM).
/// For more details, refer the paper 'RISA: Accurate and efficient placement routability modeling'
// WARNING: the 'low ... high' syntax below is a GCC extension
// It is not guaranteed to be supported by other compilers
#define DEFINE_NET_WIRING_DISTRIBUTION_MAP_WEIGHT  \
    T netWiringDistributionMapWeight(int num_pins) \
    {                                              \
        \
        switch (num_pins)                          \
        {                                          \
        case 1: case 2:  case 3:                              \
            return 1.0000;                         \
        case 4:                                    \
            return 1.0828;                         \
        case 5:                                    \
            return 1.1536;                         \
        case 6:                                    \
            return 1.2206;                         \
        case 7:                                    \
            return 1.2823;                         \
        case 8:                                    \
            return 1.3385;                         \
        case 9:                                    \
            return 1.3991;                         \
        case 10:                                   \
            return 1.4493;                         \
        case 11: case 12: case 13: case 14: case 15:                            \
            return 1.6899;                         \
        case 16: case 17: case 18: case 19: case 20:                            \
            return 1.8924;                         \
        case 21: case 22: case 23: case 24: case 25:                            \
            return 2.0743;                         \
        case 26: case 27: case 28: case 29: case 30:                            \
            return 2.2334;                         \
        case 31: case 32: case 33: case 34: case 35:                            \
            return 2.3892;                         \
        case 36: case 37: case 38: case 39: case 40:                            \
            return 2.5356;                         \
        case 41: case 42: case 43: case 44: case 45:                            \
            return 2.6625;                         \
        default:                                   \
            return 2.7933;                         \
        }                                          \
    }

#endif
