#include <cmath>

// define the domain size and the halo width
int64_t domain_size = 64;
int64_t domain_height = 60;
int64_t halo_width = 4;
int64_t count = 100;

typedef double ElementType;

#include "hadvuv5th.h"
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
  Storage3D vin = allocateStorage(sizes3D);
  Storage3D uout = allocateStorage(sizes3D);
  Storage3D vout = allocateStorage(sizes3D);
  Storage1D acrlat0 = allocateStorage(size1D);
  Storage1D acrlat1 = allocateStorage(size1D);
  Storage1D tgrlatda0 = allocateStorage(size1D);
  Storage1D tgrlatda1 = allocateStorage(size1D);
  Storage3D uatupos = allocateStorage(sizes3D);
  Storage3D vatupos = allocateStorage(sizes3D);
  Storage3D uatvpos = allocateStorage(sizes3D);
  Storage3D vatvpos = allocateStorage(sizes3D);
  Storage3D uavg = allocateStorage(sizes3D);
  Storage3D vavg = allocateStorage(sizes3D);
  Storage3D ures = allocateStorage(sizes3D);
  Storage3D vres = allocateStorage(sizes3D);

  ElementType eddlat = ldexpl(1.0, -11);
  ElementType eddlon = ldexpl(1.5, -11);

  // fillMath(1.0, 3.3, 1.5, 1.5, 2.0, 4.0, uin, domain_size, domain_height);
  // fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, utens, domain_size, domain_height);
  // fillMath(1.1, 2.0, 1.5, 2.8, 2.0, 4.1, vin, domain_size, domain_height);
  // fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, vtens, domain_size, domain_height);
  // fillMath(8.0, 9.4, 1.5, 1.7, 2.0, 3.5, ppuv, domain_size, domain_height);
  // fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, wgtfac, domain_size, domain_height);
  // fillMath(5.0, 8.0, 1.5, 7.1, 2.0, 4.3, hhl, domain_size, domain_height);
  // fillMath(3.2, 7.0, 2.5, 6.1, 3.0, 2.3, rho, domain_size, domain_height);
  // fillMath(4.5, 5.0, 2.5, 2.1, 3.0, 2.3, fx, domain_size, domain_height);

  // initValue(uout, 0.0, domain_size, domain_height);
  // initValue(vout, 0.0, domain_size, domain_height);
  auto tmp = count;
  cudaDeviceSynchronize();
  count = tmp;
  while (count--) {
    timer.start("hadvuv fullfusion");
    hadvuv_fullfusion(
            uout.cudaPtr, uout.strides[0], uout.strides[1], uout.strides[2], vout.cudaPtr,  vout.strides[0],  vout.strides[1],  vout.strides[2], uin.cudaPtr,  uin.strides[0],  uin.strides[1],  uin.strides[2], vin.cudaPtr,  vin.strides[0],  vin.strides[1],  vin.strides[2],
            acrlat0.cudaPtr, acrlat1.cudaPtr, tgrlatda0.cudaPtr, tgrlatda1.cudaPtr, 
uatupos.cudaPtr, uatupos.strides[0], uatupos.strides[1], uatupos.strides[2], vatupos.cudaPtr,  vatupos.strides[0],  vatupos.strides[1],  vatupos.strides[2], uatvpos.cudaPtr,  uatvpos.strides[0],  uatvpos.strides[1],  uatvpos.strides[2], vatvpos.cudaPtr,  vatvpos.strides[0],  vatvpos.strides[1],  vatvpos.strides[2], uavg.cudaPtr,  uavg.strides[0],  uavg.strides[1],  uavg.strides[2], vavg.cudaPtr,  vavg.strides[0],  vavg.strides[1],  vavg.strides[2] ,ures.cudaPtr,  ures.strides[0],  ures.strides[1],  ures.strides[2], vres.cudaPtr,  vres.strides[0],  vres.strides[1],  vres.strides[2],
            eddlat, eddlon
            );
    timer.stop("hadvuv fullfusion");
  }
  count = tmp;
  while (count--) {
    timer.start("hadvuv fullfusion2");
    hadvuv_fullfusion2(
            uout.cudaPtr, uout.strides[0], uout.strides[1], uout.strides[2], vout.cudaPtr,  vout.strides[0],  vout.strides[1],  vout.strides[2], uin.cudaPtr,  uin.strides[0],  uin.strides[1],  uin.strides[2], vin.cudaPtr,  vin.strides[0],  vin.strides[1],  vin.strides[2],
            acrlat0.cudaPtr, acrlat1.cudaPtr, tgrlatda0.cudaPtr, tgrlatda1.cudaPtr, 
uatupos.cudaPtr, uatupos.strides[0], uatupos.strides[1], uatupos.strides[2], vatupos.cudaPtr,  vatupos.strides[0],  vatupos.strides[1],  vatupos.strides[2], uatvpos.cudaPtr,  uatvpos.strides[0],  uatvpos.strides[1],  uatvpos.strides[2], vatvpos.cudaPtr,  vatvpos.strides[0],  vatvpos.strides[1],  vatvpos.strides[2], uavg.cudaPtr,  uavg.strides[0],  uavg.strides[1],  uavg.strides[2], vavg.cudaPtr,  vavg.strides[0],  vavg.strides[1],  vavg.strides[2], ures.cudaPtr,  ures.strides[0],  ures.strides[1],  ures.strides[2], vres.cudaPtr,  vres.strides[0],  vres.strides[1],  vres.strides[2],
            eddlat, eddlon
            );
    timer.stop("hadvuv fullfusion2");
  }
  // // case PARTIAL_FUSION: {
  // //   while (count--) {
  // //     timer.start("hadvuv partialfusion");
  // //     hadvuv_partialfusion(uout, vout, uin, vin, utens, vtens, wgtfac, ppuv, hhl, rho, fx, ppgk, ppgc,
  // ppgu, ppgv, edadlat, dt);
  // //     timer.stop("hadvuv partialfusion");
  // //   }
  // //   break;
  // // }
  // case OPENMP: {
  //   while (count--) {
  //     timer.start("hadvuv openmp");
  //     hadvuv_openmp(uout, vout, uin, vin, utens, vtens, wgtfac, ppuv, hhl, rho, fx, ppgk, ppgc, ppgu, ppgv,
  //     edadlat, dt); timer.stop("hadvuv openmp");
  //   }
  //   break;
  // }
  timer.show_all();


  return 0;
}
