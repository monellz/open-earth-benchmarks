#ifndef LAPLACE_H
#define LAPLACE_H

#ifndef MANUAL_FUSION
void laplace(const Storage3D& in, Storage3D& out) {
  for (int64_t i = 0; i < domain_size; ++i) {
    for (int64_t j = 0; j < domain_size; ++j) {
      for (int64_t k = 0; k < domain_height; ++k) {
        out(i, j, k) = -4.0 * in(i, j, k) + ((in(i - 1, j, k) + in(i + 1, j, k)) + (in(i, j + 1, k) + in(i, j - 1, k)));
      }
    }
  }
}

#else
void laplace(const Storage3D& in, Storage3D& out) {
  for (int64_t i = 0; i < domain_size; ++i) {
    for (int64_t j = 0; j < domain_size; ++j) {
      for (int64_t k = 0; k < domain_height; ++k) {
        out(i, j, k) = -4.0 * in(i, j, k) + ((in(i - 1, j, k) + in(i + 1, j, k)) + (in(i, j + 1, k) + in(i, j - 1, k)));
      }
    }
  }
}
#endif

#endif  // LAPLACE_H
