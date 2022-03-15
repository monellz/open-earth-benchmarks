#include <cmath>

// define the domain size and the halo width
int64_t domain_size = 64;
int64_t domain_height = 60;
int64_t halo_width = 4;
int64_t count = 100;

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
    auto tmp = count;
      while (count--) {
        timer.start("nh_p_grad fullfusion");
        nh_p_grad_fullfusion
            (uout.cudaPtr, uout.strides[2], uout.strides[1], uout.strides[0], vout.cudaPtr,  vout.strides[2],  vout.strides[1],  vout.strides[0], uin.cudaPtr,  uin.strides[2],  uin.strides[1],  uin.strides[0], vin.cudaPtr,  vin.strides[2],  vin.strides[1],  vin.strides[0], rdx.cudaPtr,  rdx.strides[2],  rdx.strides[1],  rdx.strides[0], rdy.cudaPtr,  rdy.strides[2],  rdy.strides[1],  rdy.strides[0], gz.cudaPtr,  gz.strides[2],  gz.strides[1],  gz.strides[0], pp.cudaPtr,  pp.strides[2],  pp.strides[1],  pp.strides[0], pk3.cudaPtr,  pk3.strides[2],  pk3.strides[1],  pk3.strides[0], wk1.cudaPtr,  wk1.strides[2],  wk1.strides[1],  wk1.strides[0], wk.cudaPtr,  wk.strides[2],  wk.strides[1],  wk.strides[0], du.cudaPtr,  du.strides[2],  du.strides[1],  du.strides[0], dv.cudaPtr,  dv.strides[2],  dv.strides[1],  dv.strides[0], dt);
        timer.stop("nh_p_grad fullfusion");
      }
      count = tmp;
      while (count--) {
        timer.start("nh_p_grad fullfusion2");
        nh_p_grad_fullfusion2
            (uout.cudaPtr, uout.strides[2], uout.strides[1], uout.strides[0], vout.cudaPtr,  vout.strides[2],  vout.strides[1],  vout.strides[0], uin.cudaPtr,  uin.strides[2],  uin.strides[1],  uin.strides[0], vin.cudaPtr,  vin.strides[2],  vin.strides[1],  vin.strides[0], rdx.cudaPtr,  rdx.strides[2],  rdx.strides[1],  rdx.strides[0], rdy.cudaPtr,  rdy.strides[2],  rdy.strides[1],  rdy.strides[0], gz.cudaPtr,  gz.strides[2],  gz.strides[1],  gz.strides[0], pp.cudaPtr,  pp.strides[2],  pp.strides[1],  pp.strides[0], pk3.cudaPtr,  pk3.strides[2],  pk3.strides[1],  pk3.strides[0], wk1.cudaPtr,  wk1.strides[2],  wk1.strides[1],  wk1.strides[0], wk.cudaPtr,  wk.strides[2],  wk.strides[1],  wk.strides[0], du.cudaPtr,  du.strides[2],  du.strides[1],  du.strides[0], dv.cudaPtr,  dv.strides[2],  dv.strides[1],  dv.strides[0], dt);
        timer.stop("nh_p_grad fullfusion2");
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
