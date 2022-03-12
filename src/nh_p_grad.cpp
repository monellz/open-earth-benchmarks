#include <cmath>

// define the domain size and the halo width
int64_t domain_size = 64;
int64_t domain_height = 60;
int64_t halo_width = 4;
int64_t count = 1000;

typedef double ElementType;

#include "util.h"
#include "nh_p_grad.h"

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
  Storage3D uin = allocateStorage(sizes3D);
  Storage3D vin = allocateStorage(sizes3D);
  Storage3D rdx = allocateStorage(sizes3D);
  Storage3D rdy = allocateStorage(sizes3D);
  Storage3D gz = allocateStorage(sizes3D);
  Storage3D pp = allocateStorage(sizes3D);
  Storage3D pk3 = allocateStorage(sizes3D);
  Storage3D wk1 = allocateStorage(sizes3D);
  Storage3D uout = allocateStorage(sizes3D);
  Storage3D vout = allocateStorage(sizes3D);
  Storage3D wk = allocateStorage(sizes3D);
  Storage3D du = allocateStorage(sizes3D);
  Storage3D dv = allocateStorage(sizes3D);

  ElementType dt = 0.1;

  fillMath(1.0, 3.3, 1.5, 1.5, 2.0, 4.0, uin, domain_size, domain_height);
  fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, vin, domain_size, domain_height);
  fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, rdx, domain_size, domain_height);
  fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, rdy, domain_size, domain_height);
  fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, gz, domain_size, domain_height);
  fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, pp, domain_size, domain_height);
  fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, pk3, domain_size, domain_height);
  fillMath(3.2, 7.0, 2.5, 6.1, 3.0, 2.3, wk1, domain_size, domain_height);

  initValue(uout, 0.0, domain_size, domain_height);
  initValue(vout, 0.0, domain_size, domain_height);

  switch (ALGO) {
    case DEFAULT: {
      while (count--) {
        timer.start("nh_p_grad");
        nh_p_grad(uout, vout, uin, vin, rdx, rdy, gz, pp, pk3, wk1, wk, du, dv, dt);
        timer.stop("nh_p_grad");
      }
      break;
    }
    case FULL_FUSION: {
      while (count--) {
        timer.start("nh_p_grad fullfusion");
        nh_p_grad_fullfusion(uout, vout, uin, vin, rdx, rdy, gz, pp, pk3, wk1, wk, du, dv, dt);
        timer.stop("nh_p_grad fullfusion");
      }
      break;
    }
    case PARTIAL_FUSION: {
      while (count--) {
        timer.start("nh_p_grad partialfusion");
        nh_p_grad_partialfusion(uout, vout, uin, vin, rdx, rdy, gz, pp, pk3, wk1, wk, du, dv, dt);
        timer.stop("nh_p_grad partialfusion");
      }
      break;
    }
    default: {
      std::cout << "Unknown ALGO" << std::endl;
    }
  }
  timer.show_all();

  // free the storage
  freeStorage(uin);
  freeStorage(vin);
  freeStorage(rdx);
  freeStorage(rdy);
  freeStorage(gz);
  freeStorage(pp);
  freeStorage(pk3);
  freeStorage(wk1);
  freeStorage(uout);
  freeStorage(vout);
  freeStorage(wk);
  freeStorage(du);
  freeStorage(dv);

  return 0;
}
