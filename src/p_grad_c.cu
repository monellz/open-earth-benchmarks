#include <cmath>

// define the domain size and the halo width
int64_t domain_size = 64;
int64_t domain_height = 60;
int64_t halo_width = 4;
int64_t count = 1000;

typedef double ElementType;

#include "util.h"
#include "p_grad_c.h"

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

  // allocate the storage
  Storage3D uin = allocateStorage(sizes3D);
  Storage3D vin = allocateStorage(sizes3D);
  Storage3D rdxc = allocateStorage(sizes3D);
  Storage3D rdyc = allocateStorage(sizes3D);
  Storage3D delpc = allocateStorage(sizes3D);
  Storage3D gz = allocateStorage(sizes3D);
  Storage3D pkc = allocateStorage(sizes3D);
  Storage3D uout = allocateStorage(sizes3D);
  Storage3D vout = allocateStorage(sizes3D);
  Storage3D wk = allocateStorage(sizes3D);

  ElementType dt2 = 0.1;

  fillMath(1.0, 3.3, 1.5, 1.5, 2.0, 4.0, uin, domain_size, domain_height);
  fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, vin, domain_size, domain_height);
  fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, rdxc, domain_size, domain_height);
  fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, rdyc, domain_size, domain_height);
  fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, delpc, domain_size, domain_height);
  fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, gz, domain_size, domain_height);
  fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, pkc, domain_size, domain_height);

  initValue(uout, 0.0, domain_size, domain_height);
  initValue(vout, 0.0, domain_size, domain_height);
    
  auto tmp = count;
      while (count--) {
        timer.start("p_grad_c fullfusion");
        p_grad_c_fullfusion
            (uout.cudaPtr, uout.strides[2], uout.strides[1], uout.strides[0], vout.cudaPtr,  vout.strides[2],  vout.strides[1],  vout.strides[0], uin.cudaPtr,  uin.strides[2],  uin.strides[1],  uin.strides[0], vin.cudaPtr,  vin.strides[2],  vin.strides[1],  vin.strides[0], rdxc.cudaPtr,  rdxc.strides[2],  rdxc.strides[1],  rdxc.strides[0], rdyc.cudaPtr,  rdyc.strides[2],  rdyc.strides[1],  rdyc.strides[0], delpc.cudaPtr,  delpc.strides[2],  delpc.strides[1],  delpc.strides[0], gz.cudaPtr,  gz.strides[2],  gz.strides[1],  gz.strides[0], pkc.cudaPtr,  pkc.strides[2],  pkc.strides[1],  pkc.strides[0], wk.cudaPtr,  wk.strides[2],  wk.strides[1],  wk.strides[0], dt2);
        timer.stop("p_grad_c fullfusion");
      }
    count = tmp;
      while (count--) {
        timer.start("p_grad_c fullfusion2");
        p_grad_c_fullfusion2
            (uout.cudaPtr, uout.strides[2], uout.strides[1], uout.strides[0], vout.cudaPtr,  vout.strides[2],  vout.strides[1],  vout.strides[0], uin.cudaPtr,  uin.strides[2],  uin.strides[1],  uin.strides[0], vin.cudaPtr,  vin.strides[2],  vin.strides[1],  vin.strides[0], rdxc.cudaPtr,  rdxc.strides[2],  rdxc.strides[1],  rdxc.strides[0], rdyc.cudaPtr,  rdyc.strides[2],  rdyc.strides[1],  rdyc.strides[0], delpc.cudaPtr,  delpc.strides[2],  delpc.strides[1],  delpc.strides[0], gz.cudaPtr,  gz.strides[2],  gz.strides[1],  gz.strides[0], pkc.cudaPtr,  pkc.strides[2],  pkc.strides[1],  pkc.strides[0], wk.cudaPtr,  wk.strides[2],  wk.strides[1],  wk.strides[0], dt2);
        timer.stop("p_grad_c fullfusion2");
      }
  timer.show_all();

  // free the storage
  freeStorage(uin);
  freeStorage(vin);
  freeStorage(rdxc);
  freeStorage(rdyc);
  freeStorage(delpc);
  freeStorage(gz);
  freeStorage(pkc);
  freeStorage(uout);
  freeStorage(vout);
  freeStorage(wk);

  return 0;
}
