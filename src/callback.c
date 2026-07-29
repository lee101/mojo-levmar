#include <stdint.h>

typedef void (*mojo_levmar_callback)(
    int64_t, int64_t, int64_t, int64_t, int64_t
);

void mlm_call_f64(
    int64_t fn, int64_t p, int64_t dst, int64_t m, int64_t n, int64_t data
) {
    ((mojo_levmar_callback)(uintptr_t)fn)(p, dst, m, n, data);
}

void mlm_call_f32(
    int64_t fn, int64_t p, int64_t dst, int64_t m, int64_t n, int64_t data
) {
    ((mojo_levmar_callback)(uintptr_t)fn)(p, dst, m, n, data);
}
