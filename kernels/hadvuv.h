#ifndef HADVUV_H
#define HADVUV_H

#include "util.h"

__device__ double advectionDriver(double *field,int64_t field_offset_3_3, int64_t field_offset_3_2, int64_t field_offset_3_1, const int64_t i, const int64_t j, const int64_t k,
                            const double uavg, const double vavg, const double eddlat,
                            const double eddlon) {
  double result_x = 0.0;
  double result_y = 0.0;

  if (uavg > 0) {
    result_x = uavg * ((-1.0) / (6.0) * field[(i - 2) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1] + field[(i - 1) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1] +
                       (-0.5) * field[(i) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1] + (-1.0) / (3.0) * field[(i + 1) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1]);
  } else {
    result_x = -uavg * ((-1.0) / (3.0) * field[(i - 1) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1] + (-0.5) * field[(i) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1] +
                        field[(i + 1) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1] + (-1.0) / (6.0) * field[(i + 2) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1]);
  }

  if (vavg > 0) {
    result_y = vavg * ((-1.0) / (6.0) * field[(i) * field_offset_3_3 + ( j - 2) * field_offset_3_2 + ( k) * field_offset_3_1] + field[(i) * field_offset_3_3 + ( j - 1) * field_offset_3_2 + ( k) * field_offset_3_1] +
                       (-0.5) * field[(i) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1] + (-1.0) / (3.0) * field[(i) * field_offset_3_3 + ( j + 1) * field_offset_3_2 + ( k) * field_offset_3_1]);
  } else {
    result_y = -vavg * ((-1.0) / (3.0) * field[(i) * field_offset_3_3 + ( j - 1) * field_offset_3_2 + ( k) * field_offset_3_1] + (-0.5) * field[(i) * field_offset_3_3 + ( j) * field_offset_3_2 + ( k) * field_offset_3_1] +
                        field[(i) * field_offset_3_3 + ( j + 1) * field_offset_3_2 + ( k) * field_offset_3_1] + (-1.0) / (6.0) * field[(i) * field_offset_3_3 + ( j + 2) * field_offset_3_2 + ( k) * field_offset_3_1]);
  }

  return eddlat * result_x + eddlon * result_y;
}

__global__ void hadvuv_fullfusion_k1(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, const double * acrlat0,
            const double * acrlat1, const double * tgrlatda0, const double * tgrlatda1, double *uatupos,int64_t uatupos_offset_3_3, int64_t uatupos_offset_3_2, int64_t uatupos_offset_3_1,
            double *vatupos,int64_t vatupos_offset_3_3, int64_t vatupos_offset_3_2, int64_t vatupos_offset_3_1, double *uatvpos,int64_t uatvpos_offset_3_3, int64_t uatvpos_offset_3_2, int64_t uatvpos_offset_3_1, double *vatvpos,int64_t vatvpos_offset_3_3, int64_t vatvpos_offset_3_2, int64_t vatvpos_offset_3_1, double *uavg,int64_t uavg_offset_3_3, int64_t uavg_offset_3_2, int64_t uavg_offset_3_1, double *vavg,int64_t vavg_offset_3_3, int64_t vavg_offset_3_2, int64_t vavg_offset_3_1,
            double *ures,int64_t ures_offset_3_3, int64_t ures_offset_3_2, int64_t ures_offset_3_1, double *vres,int64_t vres_offset_3_3, int64_t vres_offset_3_2, int64_t vres_offset_3_1, const double eddlat, const double eddlon, int64_t domain_size, int64_t domain_height) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size) * (domain_size) * (domain_height)) return;
        int k = tid / (domain_size) / (domain_size);
        int i = (tid % ((domain_size) * (domain_size))) / (domain_size);
        int j = tid % (domain_size);
  //for (int64_t k = 0; k < domain_height; ++k) {
  //  for (int64_t i = 0; i < domain_size; ++i) {
  //    for (int64_t j = 0; j < domain_size; ++j) {
        auto _uatupos = ((1.0) / (3.0)) * (uin[(i - 1) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] + uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] + uin[(i + 1) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1]);

        auto _vatupos =
            (0.25) * (vin[(i + 1) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] + vin[(i + 1) * vin_offset_3_3 + ( j - 1) * vin_offset_3_2 + ( k) * vin_offset_3_1] + vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] + vin[(i) * vin_offset_3_3 + ( j - 1) * vin_offset_3_2 + ( k) * vin_offset_3_1]);

        auto _uavg = acrlat0[j] * _uatupos;
        auto _vavg = DOUBLE_EARTH_RADIUS_RECIP * _vatupos;

        auto _ures =  advectionDriver(uin, uin_offset_3_3, uin_offset_3_2, uin_offset_3_1, i, j, k, _uavg, _vavg, eddlat, eddlon); //advectionDriver[(vin) * advectionDriver_offset_3_3 + ( i) * advectionDriver_offset_3_2 + ( j, k, _uavg, _vavg, eddlat, eddlon) * advectionDriver_offset_3_1];
        uout[(i) * uout_offset_3_3 + ( j) * uout_offset_3_2 + ( k) * uout_offset_3_1] = _ures + tgrlatda0[j] * uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] * _vatupos;

        auto _uatvpos =
            (0.25) * (uin[(i - 1) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] + uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] + uin[(i) * uin_offset_3_3 + ( j + 1) * uin_offset_3_2 + ( k) * uin_offset_3_1] + uin[(i - 1) * uin_offset_3_3 + ( j + 1) * uin_offset_3_2 + ( k) * uin_offset_3_1]);

        auto _vatvpos = (1.0) / (3.0) * (vin[(i) * vin_offset_3_3 + ( j - 1) * vin_offset_3_2 + ( k) * vin_offset_3_1] + vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] + vin[(i) * vin_offset_3_3 + ( j + 1) * vin_offset_3_2 + ( k) * vin_offset_3_1]);
        _uavg = acrlat1[j] * _uatvpos;
        _vavg = DOUBLE_EARTH_RADIUS_RECIP * _vatvpos;

        auto _vres =  advectionDriver(vin, vin_offset_3_3, vin_offset_3_2, vin_offset_3_1, i, j, k, _uavg, _vavg, eddlat, eddlon); //advectionDriver[(vin) * advectionDriver_offset_3_3 + ( i) * advectionDriver_offset_3_2 + ( j, k, _uavg, _vavg, eddlat, eddlon) * advectionDriver_offset_3_1];

        vout[(i) * vout_offset_3_3 + ( j) * vout_offset_3_2 + ( k) * vout_offset_3_1] = _vres - tgrlatda1[j] * _uatvpos * _uatvpos;
}

void hadvuv_fullfusion(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, const double * acrlat0,
            const double * acrlat1, const double * tgrlatda0, const double * tgrlatda1, double *uatupos,int64_t uatupos_offset_3_3, int64_t uatupos_offset_3_2, int64_t uatupos_offset_3_1,
            double *vatupos,int64_t vatupos_offset_3_3, int64_t vatupos_offset_3_2, int64_t vatupos_offset_3_1, double *uatvpos,int64_t uatvpos_offset_3_3, int64_t uatvpos_offset_3_2, int64_t uatvpos_offset_3_1, double *vatvpos,int64_t vatvpos_offset_3_3, int64_t vatvpos_offset_3_2, int64_t vatvpos_offset_3_1, double *uavg,int64_t uavg_offset_3_3, int64_t uavg_offset_3_2, int64_t uavg_offset_3_1, double *vavg,int64_t vavg_offset_3_3, int64_t vavg_offset_3_2, int64_t vavg_offset_3_1,
            double *ures,int64_t ures_offset_3_3, int64_t ures_offset_3_2, int64_t ures_offset_3_1, double *vres,int64_t vres_offset_3_3, int64_t vres_offset_3_2, int64_t vres_offset_3_1, const double eddlat, const double eddlon) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 64;
    hadvuv_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1, uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1, vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1,   acrlat0,
              acrlat1,   tgrlatda0,   tgrlatda1, uatupos, uatupos_offset_3_3,  uatupos_offset_3_2,  uatupos_offset_3_1,
            vatupos, vatupos_offset_3_3,  vatupos_offset_3_2,  vatupos_offset_3_1, uatvpos, uatvpos_offset_3_3,  uatvpos_offset_3_2,  uatvpos_offset_3_1, vatvpos, vatvpos_offset_3_3,  vatvpos_offset_3_2,  vatvpos_offset_3_1, uavg, uavg_offset_3_3,  uavg_offset_3_2,  uavg_offset_3_1, vavg, vavg_offset_3_3,  vavg_offset_3_2,  vavg_offset_3_1,
            ures, ures_offset_3_3,  ures_offset_3_2,  ures_offset_3_1, vres, vres_offset_3_3,  vres_offset_3_2,  vres_offset_3_1,   eddlat,   eddlon, domain_size, domain_height);
}

void hadvuv_fullfusion2(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, const double * acrlat0,
            const double * acrlat1, const double * tgrlatda0, const double * tgrlatda1, double *uatupos,int64_t uatupos_offset_3_3, int64_t uatupos_offset_3_2, int64_t uatupos_offset_3_1,
            double *vatupos,int64_t vatupos_offset_3_3, int64_t vatupos_offset_3_2, int64_t vatupos_offset_3_1, double *uatvpos,int64_t uatvpos_offset_3_3, int64_t uatvpos_offset_3_2, int64_t uatvpos_offset_3_1, double *vatvpos,int64_t vatvpos_offset_3_3, int64_t vatvpos_offset_3_2, int64_t vatvpos_offset_3_1, double *uavg,int64_t uavg_offset_3_3, int64_t uavg_offset_3_2, int64_t uavg_offset_3_1, double *vavg,int64_t vavg_offset_3_3, int64_t vavg_offset_3_2, int64_t vavg_offset_3_1,
            double *ures,int64_t ures_offset_3_3, int64_t ures_offset_3_2, int64_t ures_offset_3_1, double *vres,int64_t vres_offset_3_3, int64_t vres_offset_3_2, int64_t vres_offset_3_1, const double eddlat, const double eddlon) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 512;
    hadvuv_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1, uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1, vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1,   acrlat0,
              acrlat1,   tgrlatda0,   tgrlatda1, uatupos, uatupos_offset_3_3,  uatupos_offset_3_2,  uatupos_offset_3_1,
            vatupos, vatupos_offset_3_3,  vatupos_offset_3_2,  vatupos_offset_3_1, uatvpos, uatvpos_offset_3_3,  uatvpos_offset_3_2,  uatvpos_offset_3_1, vatvpos, vatvpos_offset_3_3,  vatvpos_offset_3_2,  vatvpos_offset_3_1, uavg, uavg_offset_3_3,  uavg_offset_3_2,  uavg_offset_3_1, vavg, vavg_offset_3_3,  vavg_offset_3_2,  vavg_offset_3_1,
            ures, ures_offset_3_3,  ures_offset_3_2,  ures_offset_3_1, vres, vres_offset_3_3,  vres_offset_3_2,  vres_offset_3_1,   eddlat,   eddlon, domain_size, domain_height);
}

#endif  // HADVUV_H
