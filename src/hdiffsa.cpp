#include <cmath>

// define the domain size and the halo width
int64_t domain_size = 64;
int64_t domain_height = 60;
int64_t halo_width = 4;
int64_t count = 1000;

typedef double ElementType;

#include "util.h"
#include "hdiffsa.h"

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

  const int64_t size1D = domain_size + 2 * halo_width;
  const std::array<int64_t, 3> sizes3D = {domain_size + 2 * halo_width, domain_size + 2 * halo_width,
                                          domain_height + 2 * halo_width};

  Storage3D in = allocateStorage(sizes3D);
  Storage3D mask = allocateStorage(sizes3D);
  Storage3D out = allocateStorage(sizes3D);
  Storage1D crlato = allocateStorage(size1D);
  Storage1D crlatu = allocateStorage(size1D);
  Storage3D lap = allocateStorage(sizes3D);
  Storage3D flx = allocateStorage(sizes3D);
  Storage3D fly = allocateStorage(sizes3D);

  fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, in, domain_size, domain_height);
  fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, crlato, domain_size, domain_height);
  fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, crlatu, domain_size, domain_height);
  initValue(mask, 0.025, domain_size, domain_height);
  initValue(out, 0.0, domain_size, domain_height);

  std::cout << "-> starting verification" << std::endl;

  switch (ALGO) {
    case DEFAULT: {
      while (count--) {
        timer.start("hdiffsa");
        hdiffsa(in, mask, out, crlato, crlatu, lap, flx, fly);
        timer.stop("hdiffsa");
      }
      break;
    }
    case FULL_FUSION: {
      while (count--) {
        timer.start("hdiffsa fullfusion");
        hdiffsa_fullfusion(in, mask, out, crlato, crlatu, lap, flx, fly);
        timer.stop("hdiffsa fullfusion");
      }
      break;
    }
    case PARTIAL_FUSION: {
      while (count--) {
        timer.start("hdiffsa partialfusion");
        hdiffsa_partialfusion(in, mask, out, crlato, crlatu, lap, flx, fly);
        timer.stop("hdiffsa partialfusion");
      }
      break;
    }
    case OPENMP: {
      while (count--) {
        timer.start("hdiffsa openmp");
        hdiffsa_openmp(in, mask, out, crlato, crlatu, lap, flx, fly);
        timer.stop("hdiffsa openmp");
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
  freeStorage(mask);
  freeStorage(out);
  freeStorage(crlato);
  freeStorage(crlatu);
  freeStorage(lap);
  freeStorage(flx);
  freeStorage(fly);

  return 0;
}
