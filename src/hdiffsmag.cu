#include <cmath>

// define the domain size and the halo width
int64_t domain_size = 64;
int64_t domain_height = 60;
int64_t halo_width = 4;
int64_t count = 1000;

typedef double ElementType;

#include "util.h"
#include "hdiffsmag.h"

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

  Storage3D uin = allocateStorage(sizes3D);
  Storage3D vin = allocateStorage(sizes3D);
  Storage3D mask = allocateStorage(sizes3D);
  Storage3D uout = allocateStorage(sizes3D);
  Storage3D vout = allocateStorage(sizes3D);
  Storage1D crlavo = allocateStorage(size1D);
  Storage1D crlavu = allocateStorage(size1D);
  Storage1D crlato = allocateStorage(size1D);
  Storage1D crlatu = allocateStorage(size1D);
  Storage1D acrlat0 = allocateStorage(size1D);
  Storage3D T_sqr_s = allocateStorage(sizes3D);
  Storage3D S_sqr_uv = allocateStorage(sizes3D);

  ElementType eddlat = ldexpl(1.0, -11);
  ElementType eddlon = ldexpl(1.5, -11);

  ElementType tau_smag = 0.025;
  ElementType weight_smag = 0.01;

  // fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, uin, domain_size, domain_height);
  // fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, vin, domain_size, domain_height);

  // fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, crlavo, domain_size, domain_height);
  // fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, crlavu, domain_size, domain_height);
  // fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, crlato, domain_size, domain_height);
  // fillMath(3.2, 7.0, 2.5, 6.1, 3.0, 2.3, crlatu, domain_size, domain_height);
  // fillMath(4.5, 5.0, 2.5, 2.1, 3.0, 2.3, acrlat0, domain_size, domain_height);

  // initValue(mask, 0.025, domain_size, domain_height);
  // initValue(uout, 0.0, domain_size, domain_height);
  // initValue(vout, 0.0, domain_size, domain_height);

  // initValue(T_sqr_s, 0.0, domain_size, domain_height);
  // initValue(S_sqr_uv, 0.0, domain_size, domain_height);
    auto tmp = count;
      while (count--) {
        timer.start("hdiffsmag_fullfusion2");
        hdiffsmag_fullfusion2
            (
                uout.cudaPtr, uout.strides[0], uout.strides[1], uout.strides[2], vout.cudaPtr,  vout.strides[0],  vout.strides[1],  vout.strides[2], uin.cudaPtr,  uin.strides[0],  uin.strides[1],  uin.strides[2], vin.cudaPtr,  vin.strides[0],  vin.strides[1],  vin.strides[2], mask.cudaPtr,  mask.strides[0],  mask.strides[1],  mask.strides[2],
                crlavo.cudaPtr, crlavu.cudaPtr, crlato.cudaPtr, crlatu.cudaPtr, acrlat0.cudaPtr, 
                T_sqr_s.cudaPtr, T_sqr_s.strides[0], T_sqr_s.strides[1], T_sqr_s.strides[2], S_sqr_uv.cudaPtr,  S_sqr_uv.strides[0],  S_sqr_uv.strides[1],  S_sqr_uv.strides[2],
                eddlat, eddlon,
                  tau_smag, weight_smag);
        timer.stop("hdiffsmag_fullfusion2");
      }
      count = tmp;
      while (count--) {
        timer.start("hdiffsmag fullfusion");
        hdiffsmag_fullfusion
            (
                uout.cudaPtr, uout.strides[0], uout.strides[1], uout.strides[2], vout.cudaPtr,  vout.strides[0],  vout.strides[1],  vout.strides[2], uin.cudaPtr,  uin.strides[0],  uin.strides[1],  uin.strides[2], vin.cudaPtr,  vin.strides[0],  vin.strides[1],  vin.strides[2], mask.cudaPtr,  mask.strides[0],  mask.strides[1],  mask.strides[2],
                crlavo.cudaPtr, crlavu.cudaPtr, crlato.cudaPtr, crlatu.cudaPtr, acrlat0.cudaPtr, 
                T_sqr_s.cudaPtr, T_sqr_s.strides[0], T_sqr_s.strides[1], T_sqr_s.strides[2], S_sqr_uv.cudaPtr,  S_sqr_uv.strides[0],  S_sqr_uv.strides[1],  S_sqr_uv.strides[2],
                eddlat, eddlon,
                  tau_smag, weight_smag);
        timer.stop("hdiffsmag fullfusion");
      }
  timer.show_all();

  // free the storage
  freeStorage(uin);
  freeStorage(vin);
  freeStorage(mask);
  freeStorage(uout);
  freeStorage(vout);
  freeStorage(crlavo);
  freeStorage(crlavu);
  freeStorage(crlato);
  freeStorage(crlatu);
  freeStorage(acrlat0);
  freeStorage(T_sqr_s);
  freeStorage(S_sqr_uv);

  return 0;
}
