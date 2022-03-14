#ifndef UTIL_H
#define UTIL_H

#include <assert.h>
#include <sys/time.h>

#include <array>
#include <iostream>
#include <unordered_map>

#define DEFAULT 0
#define FULL_FUSION 1
#define PARTIAL_FUSION 2
#define THREAD_MAP 2
#define OPENMP 3

#define get_elapsed_time_ms(_s, _e) (1000.0 * (_e.tv_sec - _s.tv_sec) + (_e.tv_usec - _s.tv_usec) / 1000.0)

#define EARTH_RADIUS ((ElementType)6371.229e3)  // radius of the earth
#define EARTH_RADIUS_RECIP ((ElementType)1.0 / EARTH_RADIUS)

const ElementType pi(std::acos(-1.0));

namespace {

template <typename ElementType, size_t Rank>
struct Storage {
  ElementType *allocatedPtr;
  ElementType *alignedPtr;
  double *cudaPtr = nullptr;
  int64_t allocSize;
  int64_t offset;
  std::array<int64_t, Rank> sizes;    // omitted when rank == 0
  std::array<int64_t, Rank> strides;  // omitted when rank == 0

  template <typename... T>
  struct type {};

  template <typename... T>
  ElementType &operator()(T... arg) {
    return operator()(type<T...>(), arg...);
  }

  void check() { assert(cudaPtr != nullptr); }
  void display() {
    printf("===\n");
    printf("size %lld\n", allocSize);
    for (auto &item : strides) printf("%lld\n", item);
    printf("===\n");
  }

  template <typename... T>
  const ElementType &operator()(T... arg) const {
    return operator()(type<T...>(), arg...);
  }

  template <typename... T>
  ElementType &operator()(type<T...>, T... arg);

  template <typename... T>
  const ElementType &operator()(type<T...>, T... arg) const;

  void tocuda() { cudaMemcpy(cudaPtr, (double *)alignedPtr, allocSize * sizeof(double), cudaMemcpyDefault); }
};

typedef Storage<ElementType, 1> Storage1D;
typedef Storage<ElementType, 2> Storage2D;
typedef Storage<ElementType, 3> Storage3D;

template <>
template <>
ElementType &Storage<ElementType, 1>::operator()<int64_t>(Storage::type<int64_t>, int64_t i) {
  return alignedPtr[offset + strides[0] * i];
}

template <>
template <>
const ElementType &Storage<ElementType, 1>::operator()<int64_t>(Storage::type<int64_t>, int64_t i) const {
  return alignedPtr[offset + strides[0] * i];
}

template <>
template <>
ElementType &Storage<ElementType, 2>::operator()<int64_t, int64_t>(Storage::type<int64_t, int64_t>, int64_t i,
                                                                   int64_t j) {
  return alignedPtr[offset + strides[1] * i + strides[0] * j];
}

template <>
template <>
const ElementType &Storage<ElementType, 2>::operator()<int64_t, int64_t>(Storage::type<int64_t, int64_t>, int64_t i,
                                                                         int64_t j) const {
  return alignedPtr[offset + strides[1] * i + strides[0] * j];
}

template <>
template <>
ElementType &Storage<ElementType, 3>::operator()<int64_t, int64_t, int64_t>(Storage::type<int64_t, int64_t, int64_t>,
                                                                            int64_t i, int64_t j, int64_t k) {
  return alignedPtr[offset + strides[2] * i + strides[1] * j + strides[0] * k];
}

template <>
template <>
const ElementType &Storage<ElementType, 3>::operator()<int64_t, int64_t, int64_t>(
    Storage::type<int64_t, int64_t, int64_t>, int64_t i, int64_t j, int64_t k) const {
  return alignedPtr[offset + strides[2] * i + strides[1] * j + strides[0] * k];
}

}  // namespace

// allocate Storages
Storage1D allocateStorage(const int64_t size) {
  Storage1D result;
  // initialize the size
  result.sizes[0] = size;
  // initialize the strides
  result.strides[0] = 1;
  result.offset = halo_width * result.strides[0];
  result.allocatedPtr = new ElementType[size + (32 - halo_width)];
  result.alignedPtr = &result.allocatedPtr[(32 - halo_width)];
  result.allocSize = size;
  cudaMalloc(&result.cudaPtr, (result.allocSize + (32 - halo_width) * 2) * sizeof(double) * 4);
  result.cudaPtr += (32 - halo_width) + result.allocSize * 2;
  // result.display();
  return result;
}

Storage2D allocateStorage(const std::array<int64_t, 2> sizes) {
  Storage2D result;
  // initialize the size
  result.sizes[1] = sizes[0];
  result.sizes[0] = sizes[1];
  // initialize the strides
  result.strides[1] = 1;
  result.strides[0] = result.sizes[1];
  result.offset = halo_width * result.strides[0] + halo_width * result.strides[1];
  result.allocSize = sizes[0] * sizes[1];
  result.allocatedPtr = new ElementType[result.allocSize + (32 - halo_width)];
  result.alignedPtr = &result.allocatedPtr[(32 - halo_width)];
  result.allocSize = sizes[0] * sizes[1];
  cudaMalloc(&result.cudaPtr, (result.allocSize + (32 - halo_width) * 2) * sizeof(double) * 4);
  result.cudaPtr += (32 - halo_width) + result.allocSize * 2;
  // result.display();
  return result;
}

Storage3D allocateStorage(const std::array<int64_t, 3> sizes) {
  Storage3D result;
  // initialize the size
  result.sizes[2] = sizes[0];
  result.sizes[1] = sizes[1];
  result.sizes[0] = sizes[2];
  // initialize the strides
  result.strides[2] = 1;
  result.strides[1] = result.sizes[2];
  result.strides[0] = result.sizes[2] * result.sizes[1];
  result.offset = halo_width * result.strides[0] + halo_width * result.strides[1] + halo_width * result.strides[2];
  result.allocSize = sizes[0] * sizes[1] * sizes[2];
  result.allocatedPtr = new ElementType[result.allocSize + (32 - halo_width)];
  result.alignedPtr = &result.allocatedPtr[(32 - halo_width)];
  cudaMalloc(&result.cudaPtr, (result.allocSize + (32 - halo_width) * 2) * sizeof(double) * 4);
  result.cudaPtr += (32 - halo_width) + result.allocSize * 2;
  // cudaMemset(&result.cudaPtr, result.allocSize * sizeof(double));
  // result.display();
  return result;
}

template <typename Storage>
void freeStorage(Storage &ref) {
  delete ref.allocatedPtr;
  ref.allocatedPtr = nullptr;
  ref.alignedPtr = nullptr;
}

void fillMath(ElementType a, ElementType b, ElementType c, ElementType d, ElementType e, ElementType f,
              Storage3D &field, const int64_t domain_size, const int64_t domain_height) {
  ElementType dx = ElementType(1.0) / (ElementType)(domain_size + 2 * halo_width);
  ElementType dy = ElementType(1.0) / (ElementType)(domain_size + 2 * halo_width);

  for (int64_t j = -halo_width; j < domain_size + halo_width; j++) {
    for (int64_t i = -halo_width; i < domain_size + halo_width; i++) {
      ElementType x = dx * (ElementType)i;
      ElementType y = dy * (ElementType)j;
      for (int64_t k = 0; k < domain_height; k++) {
        field(i, j, k) = k * ElementType(10e-3) + a * (b + cos(pi * (x + c * y)) + sin(d * pi * (x + e * y))) / f;
      }
    }
  }
  field.tocuda();
}

void fillMath(ElementType a, ElementType b, ElementType c, ElementType d, ElementType e, ElementType f,
              Storage2D &field, const int64_t domain_size, const int64_t domain_heigh) {
  ElementType dx = ElementType(1.0) / (ElementType)(domain_size + 2 * halo_width);
  ElementType dy = ElementType(1.0) / (ElementType)(domain_size + 2 * halo_width);

  for (int64_t j = -halo_width; j < domain_size + halo_width; j++) {
    for (int64_t i = -halo_width; i < domain_size + halo_width; i++) {
      ElementType x = dx * (ElementType)i;
      ElementType y = dy * (ElementType)j;
      field(i, j) = a * (b + cos(pi * (x + c * y)) + sin(d * pi * (x + e * y))) / f;
    }
  }
  field.tocuda();
}

void fillMath(ElementType a, ElementType b, ElementType c, ElementType d, ElementType e, ElementType f,
              Storage1D &field, const int64_t domain_size, const int64_t domain_heigh) {
  ElementType dx = ElementType(1.0) / (ElementType)(domain_size + 2 * halo_width);

  for (int64_t i = -halo_width; i < domain_size + halo_width; i++) {
    ElementType x = dx * (ElementType)i;
    field(i) = a * (b + cos(pi * (c * x)) + sin(d * pi * (e * x))) / f;
  }
  field.tocuda();
}

void initValue(Storage3D &ref, const ElementType val, const int64_t domain_size, const int64_t domain_heigh) {
  for (int64_t i = -halo_width; i < domain_size + halo_width; ++i)
    for (int64_t j = -halo_width; j < domain_size + halo_width; ++j)
      for (int64_t k = -halo_width; k < domain_height + halo_width; ++k) {
        ref(i, j, k) = val;
      }
  ref.tocuda();
}

struct mytimer_t {
  std::unordered_map<const char *, double> elapsed_time_ms;
  std::unordered_map<const char *, int> count;
  std::unordered_map<const char *, timeval> time_point;

  void start(const char *func) {
    cudaDeviceSynchronize();
    timeval s;
    gettimeofday(&s, NULL);
    time_point[func] = s;
  }
  void stop(const char *func) {
    cudaDeviceSynchronize();
    timeval e;
    gettimeofday(&e, NULL);
    count[func]++;
    timeval s = time_point[func];
    elapsed_time_ms[func] += get_elapsed_time_ms(s, e);
  }
  void show_all() {
    for (auto it = elapsed_time_ms.begin(); it != elapsed_time_ms.end(); ++it) {
      auto func = it->first;
      double t = it->second;
      int c = count[func];
      printf("%s: %lf ms / %d count, avg %lf ms\n", func, t, c, t / c);
    }
  }
};

mytimer_t timer;

#endif  // UTIL_H
