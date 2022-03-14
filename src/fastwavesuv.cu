#include <cmath>

// define the domain size and the halo width
int64_t domain_size = 64;
int64_t domain_height = 60;
int64_t halo_width = 4;
int64_t count = 1000;

typedef double ElementType;

#include "fastwavesuv.h"
#include "util.h"

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
  Storage3D utens = allocateStorage(sizes3D);
  Storage3D vin = allocateStorage(sizes3D);
  Storage3D vtens = allocateStorage(sizes3D);
  Storage3D wgtfac = allocateStorage(sizes3D);
  Storage3D ppuv = allocateStorage(sizes3D);
  Storage3D hhl = allocateStorage(sizes3D);
  Storage3D rho = allocateStorage(sizes3D);
  Storage3D uout = allocateStorage(sizes3D);
  Storage3D vout = allocateStorage(sizes3D);
  Storage1D fx = allocateStorage(size1D);
  Storage3D ppgk = allocateStorage(sizes3D);
  Storage3D ppgc = allocateStorage(sizes3D);
  Storage3D ppgu = allocateStorage(sizes3D);
  Storage3D ppgv = allocateStorage(sizes3D);

  ElementType dt = 10.0;
  ElementType edadlat = ldexpl(1.0, -11);

  fillMath(1.0, 3.3, 1.5, 1.5, 2.0, 4.0, uin, domain_size, domain_height);
  fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, utens, domain_size, domain_height);
  fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, vin, domain_size, domain_height);
  fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, vtens, domain_size, domain_height);
  fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, ppuv, domain_size, domain_height);
  fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, wgtfac, domain_size, domain_height);
  fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, hhl, domain_size, domain_height);
  fillMath(3.2, 7.0, 2.5, 6.1, 3.0, 2.3, rho, domain_size, domain_height);
  fillMath(4.5, 5.0, 2.5, 2.1, 3.0, 2.3, fx, domain_size, domain_height);

  initValue(uout, 0.0, domain_size, domain_height);
  initValue(vout, 0.0, domain_size, domain_height);

  switch (ALGO) {
    case DEFAULT: {
      while (count--) {
        timer.start("fastwavesuv");
        fastwavesuv(
            // uout, vout, uin, vin, utens, vtens, wgtfac, ppuv, hhl, rho, fx, ppgk, ppgc, ppgu, ppgv,
            uout.cudaPtr, uout.strides[2], uout.strides[1], uout.strides[0], vout.cudaPtr, vout.strides[2],
            vout.strides[1], vout.strides[0], uin.cudaPtr, uin.strides[2], uin.strides[1], uin.strides[0], vin.cudaPtr,
            vin.strides[2], vin.strides[1], vin.strides[0], utens.cudaPtr, utens.strides[2], utens.strides[1],
            utens.strides[0], vtens.cudaPtr, vtens.strides[2], vtens.strides[1], vtens.strides[0], wgtfac.cudaPtr,
            wgtfac.strides[2], wgtfac.strides[1], wgtfac.strides[0], ppuv.cudaPtr, ppuv.strides[2], ppuv.strides[1],
            ppuv.strides[0], hhl.cudaPtr, hhl.strides[2], hhl.strides[1], hhl.strides[0], rho.cudaPtr, rho.strides[2],
            rho.strides[1], rho.strides[0], fx.cudaPtr, ppgk.cudaPtr, ppgk.strides[2], ppgk.strides[1], ppgk.strides[0],
            ppgc.cudaPtr, ppgc.strides[2], ppgc.strides[1], ppgc.strides[0], ppgu.cudaPtr, ppgu.strides[2],
            ppgu.strides[1], ppgu.strides[0], ppgv.cudaPtr, ppgv.strides[2], ppgv.strides[1], ppgv.strides[0],
            (double)edadlat, (double)dt);
        timer.stop("fastwavesuv");
      }
      break;
    }
    case FULL_FUSION: {
      while (count--) {
        timer.start("fastwavesuv fullfusion");
        fastwavesuv_fullfusion(
            uout.cudaPtr, uout.strides[2], uout.strides[1], uout.strides[0], vout.cudaPtr, vout.strides[2],
            vout.strides[1], vout.strides[0], uin.cudaPtr, uin.strides[2], uin.strides[1], uin.strides[0], vin.cudaPtr,
            vin.strides[2], vin.strides[1], vin.strides[0], utens.cudaPtr, utens.strides[2], utens.strides[1],
            utens.strides[0], vtens.cudaPtr, vtens.strides[2], vtens.strides[1], vtens.strides[0], wgtfac.cudaPtr,
            wgtfac.strides[2], wgtfac.strides[1], wgtfac.strides[0], ppuv.cudaPtr, ppuv.strides[2], ppuv.strides[1],
            ppuv.strides[0], hhl.cudaPtr, hhl.strides[2], hhl.strides[1], hhl.strides[0], rho.cudaPtr, rho.strides[2],
            rho.strides[1], rho.strides[0], fx.cudaPtr, ppgk.cudaPtr, ppgk.strides[2], ppgk.strides[1], ppgk.strides[0],
            ppgc.cudaPtr, ppgc.strides[2], ppgc.strides[1], ppgc.strides[0], ppgu.cudaPtr, ppgu.strides[2],
            ppgu.strides[1], ppgu.strides[0], ppgv.cudaPtr, ppgv.strides[2], ppgv.strides[1], ppgv.strides[0],
            (double)edadlat, (double)dt);
        timer.stop("fastwavesuv fullfusion");
      }
      break;
    }
    // // case PARTIAL_FUSION: {
    // //   while (count--) {
    // //     timer.start("fastwavesuv partialfusion");
    // //     fastwavesuv_partialfusion(uout, vout, uin, vin, utens, vtens, wgtfac, ppuv, hhl, rho, fx, ppgk, ppgc,
    // ppgu, ppgv, edadlat, dt);
    // //     timer.stop("fastwavesuv partialfusion");
    // //   }
    // //   break;
    // // }
    // case OPENMP: {
    //   while (count--) {
    //     timer.start("fastwavesuv openmp");
    //     fastwavesuv_openmp(uout, vout, uin, vin, utens, vtens, wgtfac, ppuv, hhl, rho, fx, ppgk, ppgc, ppgu, ppgv,
    //     edadlat, dt); timer.stop("fastwavesuv openmp");
    //   }
    //   break;
    // }
    default: {
      std::cout << "Unknown ALGO" << std::endl;
    }
  }
  timer.show_all();

  // free the storage
  freeStorage(uin);
  freeStorage(utens);
  freeStorage(vin);
  freeStorage(vtens);
  freeStorage(wgtfac);
  freeStorage(ppuv);
  freeStorage(hhl);
  freeStorage(rho);
  freeStorage(uout);
  freeStorage(vout);
  freeStorage(fx);
  freeStorage(ppgk);
  freeStorage(ppgc);
  freeStorage(ppgu);
  freeStorage(ppgv);

  return 0;
}
