#ifndef _r123array_h_
#define _r123array_h_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Array structures for Philox
struct r123array1x32 {
    uint32_t v[1];
};

struct r123array2x32 {
    uint32_t v[2];
};

struct r123array4x32 {
    uint32_t v[4];
};

#ifdef __cplusplus
}
#endif

#endif
