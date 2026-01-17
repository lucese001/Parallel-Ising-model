#ifndef _r123_features_compilerfeatures_h_
#define _r123_features_compilerfeatures_h_

// Minimal feature detection for gcc/clang on x86_64 (ALMA 9)
#define R123_USE_64BIT 1
#define R123_USE_PHILOX_64BIT 0  // Use 32-bit version for speed
#define R123_USE_MULHILO32_ASM 0
#define R123_STATIC_INLINE static inline
#define R123_FORCE_INLINE(x) x __attribute__((always_inline))
#define R123_CUDA_DEVICE
#define R123_METAL_THREAD_ADDRESS_SPACE
#define R123_METAL_CONSTANT_ADDRESS_SPACE
#define R123_STATIC_ASSERT(expr, msg) typedef char static_assertion[(expr)?1:-1]

#endif
