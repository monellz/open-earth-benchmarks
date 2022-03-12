#include <cmath>

// define the domain size and the halo width
int64_t domain_size = 64;
int64_t domain_height = 60;
int64_t halo_width = 4;
int64_t count = 1000;

typedef double ElementType;

#include "util.h"
#include "laplace.h"

// program times the execution of the linked program and times the result
int main(int argc, char **argv) {
  int ALGO = DEFAULT;
  if (argc == 4) {
    ALGO = atoi(argv[1]);
    domain_size = atoi(argv[2]);
    domain_height = atoi(argv[3]);
  } else if (argc == 2) {
    ALGO = atoi(argv[1]);
  } else if (argc != 1) {
    std::cout << "Usage: ./kernel ${ALGO} ${domain_size} ${domain_height}" << std::endl;
    exit(1);
  }
  std::cout << "ALGO: " << ALGO << std::endl;
  std::cout << "domain_size: " << domain_size << std::endl;
  std::cout << "domain_height: " << domain_height << std::endl;

  const std::array<int64_t, 3> sizes3D = {domain_size + 2 * halo_width, domain_size + 2 * halo_width,
                                          domain_height + 2 * halo_width};

  // allocate the storage
  Storage3D in = allocateStorage(sizes3D);
  Storage3D out = allocateStorage(sizes3D);
  fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, in, domain_size, domain_height);
  initValue(out, 0.0, domain_size, domain_height);

  // computing the reference version
  switch (ALGO) {
    case DEFAULT: {
      while (count--) {
        timer.start("laplace");
        laplace(in, out);
        timer.stop("laplace");
      }
      break;
    }
    case FULL_FUSION: {
      while (count--) {
        timer.start("laplace fullfusion");
        laplace_fullfusion(in, out);
        timer.stop("laplace fullfusion");
      }
      break;
    }
    case PARTIAL_FUSION: {
      while (count--) {
        timer.start("laplace partialfusion");
        laplace_partialfusion(in, out);
        timer.stop("laplace partialfusion");
      }
      break;
    }
    case OPENMP: {
      while (count--) {
        timer.start("laplace openmp");
        laplace_openmp(in, out);
        timer.stop("laplace openmp");
      }
      break;
    }
    default: {
      std::cout << "Unknown ALGO" << std::endl;
    }
  }
  timer.show_all();

  // free the storage
  freeStorage(in);
  freeStorage(out);

  return 0;
}
