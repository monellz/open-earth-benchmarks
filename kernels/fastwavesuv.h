#ifndef FASTWAVESUV_H
#define FASTWAVESUV_H

#include <stdio.h>

__global__ void fastwavesuv_k1(
        double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, double *utens,int64_t utens_offset_3_3, int64_t utens_offset_3_2, int64_t utens_offset_3_1,
                 double *vtens,int64_t vtens_offset_3_3, int64_t vtens_offset_3_2, int64_t vtens_offset_3_1, double *wgtfac,int64_t wgtfac_offset_3_3, int64_t wgtfac_offset_3_2, int64_t wgtfac_offset_3_1, double *ppuv,int64_t ppuv_offset_3_3, int64_t ppuv_offset_3_2, int64_t ppuv_offset_3_1, double *hhl,int64_t hhl_offset_3_3, int64_t hhl_offset_3_2, int64_t hhl_offset_3_1,
                 double *rho,int64_t rho_offset_3_3, int64_t rho_offset_3_2, int64_t rho_offset_3_1, double* fx, double *ppgk,int64_t ppgk_offset_3_3, int64_t ppgk_offset_3_2, int64_t ppgk_offset_3_1, double *ppgc,int64_t ppgc_offset_3_3, int64_t ppgc_offset_3_2, int64_t ppgc_offset_3_1, double *ppgu,int64_t ppgu_offset_3_3, int64_t ppgu_offset_3_2, int64_t ppgu_offset_3_1,
                 double *ppgv,int64_t ppgv_offset_3_3, int64_t ppgv_offset_3_2, int64_t ppgv_offset_3_1, double edadlat, double dt, int64_t domain_size, int64_t domain_height) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size + 1) * (domain_size + 1) * (domain_height + 1)) return;
        int i = tid / (domain_size + 1) / (domain_height + 1);
        int j = (tid % ((domain_size + 1) * (domain_height + 1))) / (domain_height + 1); 
        int k = tid % (domain_height + 1);

        // ppgk[i * 5184 + j * 72 + k] = wgtfac[i * 5184 + j * 72 + k] + ppuv[i * 5184 + j * 72 + k];
        // printf("%lld %lld %lld\n", ppgk_offset_3_3, ppgk_offset_3_2, ppgk_offset_3_1);
        ppgk[(i) * ppgk_offset_3_3 + ( j) * ppgk_offset_3_2 + ( k) * ppgk_offset_3_1] = wgtfac[(i) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k) * wgtfac_offset_3_1] * ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1] + ((1.0) - wgtfac[(i) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k) * wgtfac_offset_3_1]) * ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1];
}

__global__ void fastwavesuv_k2(
        double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, double *utens,int64_t utens_offset_3_3, int64_t utens_offset_3_2, int64_t utens_offset_3_1,
                 double *vtens,int64_t vtens_offset_3_3, int64_t vtens_offset_3_2, int64_t vtens_offset_3_1, double *wgtfac,int64_t wgtfac_offset_3_3, int64_t wgtfac_offset_3_2, int64_t wgtfac_offset_3_1, double *ppuv,int64_t ppuv_offset_3_3, int64_t ppuv_offset_3_2, int64_t ppuv_offset_3_1, double *hhl,int64_t hhl_offset_3_3, int64_t hhl_offset_3_2, int64_t hhl_offset_3_1,
                 double *rho,int64_t rho_offset_3_3, int64_t rho_offset_3_2, int64_t rho_offset_3_1, double* fx, double *ppgk,int64_t ppgk_offset_3_3, int64_t ppgk_offset_3_2, int64_t ppgk_offset_3_1, double *ppgc,int64_t ppgc_offset_3_3, int64_t ppgc_offset_3_2, int64_t ppgc_offset_3_1, double *ppgu,int64_t ppgu_offset_3_3, int64_t ppgu_offset_3_2, int64_t ppgu_offset_3_1,
                 double *ppgv,int64_t ppgv_offset_3_3, int64_t ppgv_offset_3_2, int64_t ppgv_offset_3_1, double edadlat, double dt, int64_t domain_size, int64_t domain_height) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size + 1) * (domain_size + 1) * (domain_height + 1)) return;
        int i = tid / (domain_size + 1) / (domain_height + 1);
        int j = (tid % ((domain_size + 1) * (domain_height + 1))) / (domain_height + 1); 
        int k = tid % (domain_height + 1);

        ppgc[(i) * ppgc_offset_3_3 + ( j) * ppgc_offset_3_2 + ( k) * ppgc_offset_3_1] = ppgk[(i) * ppgk_offset_3_3 + ( j) * ppgk_offset_3_2 + ( k + 1) * ppgk_offset_3_1] - ppgk[(i) * ppgk_offset_3_3 + ( j) * ppgk_offset_3_2 + ( k) * ppgk_offset_3_1];
}

__global__ void fastwavesuv_k3(
        double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, double *utens,int64_t utens_offset_3_3, int64_t utens_offset_3_2, int64_t utens_offset_3_1,
                 double *vtens,int64_t vtens_offset_3_3, int64_t vtens_offset_3_2, int64_t vtens_offset_3_1, double *wgtfac,int64_t wgtfac_offset_3_3, int64_t wgtfac_offset_3_2, int64_t wgtfac_offset_3_1, double *ppuv,int64_t ppuv_offset_3_3, int64_t ppuv_offset_3_2, int64_t ppuv_offset_3_1, double *hhl,int64_t hhl_offset_3_3, int64_t hhl_offset_3_2, int64_t hhl_offset_3_1,
                 double *rho,int64_t rho_offset_3_3, int64_t rho_offset_3_2, int64_t rho_offset_3_1, double* fx, double *ppgk,int64_t ppgk_offset_3_3, int64_t ppgk_offset_3_2, int64_t ppgk_offset_3_1, double *ppgc,int64_t ppgc_offset_3_3, int64_t ppgc_offset_3_2, int64_t ppgc_offset_3_1, double *ppgu,int64_t ppgu_offset_3_3, int64_t ppgu_offset_3_2, int64_t ppgu_offset_3_1,
                 double *ppgv,int64_t ppgv_offset_3_3, int64_t ppgv_offset_3_2, int64_t ppgv_offset_3_1, double edadlat, double dt, int64_t domain_size, int64_t domain_height) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size + 1) * (domain_size + 1) * (domain_height)) return;
        int i = tid / (domain_size + 1) / (domain_height);
        int j = (tid % ((domain_size + 1) * (domain_height))) / (domain_height); 
        int k = tid % (domain_height);

        ppgu[(i) * ppgu_offset_3_3 + ( j) * ppgu_offset_3_2 + ( k) * ppgu_offset_3_1] = (ppuv[(i + 1) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1] - ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1]) +
                        (ppgc[(i + 1) * ppgc_offset_3_3 + ( j) * ppgc_offset_3_2 + ( k) * ppgc_offset_3_1] + ppgc[(i) * ppgc_offset_3_3 + ( j) * ppgc_offset_3_2 + ( k) * ppgc_offset_3_1]) * (0.5) *
                            ((hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] + hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]) - (hhl[(i + 1) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] + hhl[(i + 1) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1])) /
                            ((hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] - hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]) + (hhl[(i + 1) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] - hhl[(i + 1) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]));

}

__global__ void fastwavesuv_k4(
        double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, double *utens,int64_t utens_offset_3_3, int64_t utens_offset_3_2, int64_t utens_offset_3_1,
                 double *vtens,int64_t vtens_offset_3_3, int64_t vtens_offset_3_2, int64_t vtens_offset_3_1, double *wgtfac,int64_t wgtfac_offset_3_3, int64_t wgtfac_offset_3_2, int64_t wgtfac_offset_3_1, double *ppuv,int64_t ppuv_offset_3_3, int64_t ppuv_offset_3_2, int64_t ppuv_offset_3_1, double *hhl,int64_t hhl_offset_3_3, int64_t hhl_offset_3_2, int64_t hhl_offset_3_1,
                 double *rho,int64_t rho_offset_3_3, int64_t rho_offset_3_2, int64_t rho_offset_3_1, double* fx, double *ppgk,int64_t ppgk_offset_3_3, int64_t ppgk_offset_3_2, int64_t ppgk_offset_3_1, double *ppgc,int64_t ppgc_offset_3_3, int64_t ppgc_offset_3_2, int64_t ppgc_offset_3_1, double *ppgu,int64_t ppgu_offset_3_3, int64_t ppgu_offset_3_2, int64_t ppgu_offset_3_1,
                 double *ppgv,int64_t ppgv_offset_3_3, int64_t ppgv_offset_3_2, int64_t ppgv_offset_3_1, double edadlat, double dt, int64_t domain_size, int64_t domain_height) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size) * (domain_size) * (domain_height)) return;
        int i = tid / (domain_size) / (domain_height);
        int j = (tid % ((domain_size) * (domain_height))) / (domain_height); 
        int k = tid % (domain_height);

        ppgv[(i) * ppgv_offset_3_3 + ( j) * ppgv_offset_3_2 + ( k) * ppgv_offset_3_1] = (ppuv[(i) * ppuv_offset_3_3 + ( j + 1) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1] - ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1]) +
                        (ppgc[(i) * ppgc_offset_3_3 + ( j + 1) * ppgc_offset_3_2 + ( k) * ppgc_offset_3_1] + ppgc[(i) * ppgc_offset_3_3 + ( j) * ppgc_offset_3_2 + ( k) * ppgc_offset_3_1]) * (0.5) *
                            ((hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] + hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]) - (hhl[(i) * hhl_offset_3_3 + ( j + 1) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] + hhl[(i) * hhl_offset_3_3 + ( j + 1) * hhl_offset_3_2 + ( k) * hhl_offset_3_1])) /
                            ((hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] - hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]) + (hhl[(i) * hhl_offset_3_3 + ( j + 1) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] - hhl[(i) * hhl_offset_3_3 + ( j + 1) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]));

}

__global__ void fastwavesuv_k5(
        double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, double *utens,int64_t utens_offset_3_3, int64_t utens_offset_3_2, int64_t utens_offset_3_1,
                 double *vtens,int64_t vtens_offset_3_3, int64_t vtens_offset_3_2, int64_t vtens_offset_3_1, double *wgtfac,int64_t wgtfac_offset_3_3, int64_t wgtfac_offset_3_2, int64_t wgtfac_offset_3_1, double *ppuv,int64_t ppuv_offset_3_3, int64_t ppuv_offset_3_2, int64_t ppuv_offset_3_1, double *hhl,int64_t hhl_offset_3_3, int64_t hhl_offset_3_2, int64_t hhl_offset_3_1,
                 double *rho,int64_t rho_offset_3_3, int64_t rho_offset_3_2, int64_t rho_offset_3_1, double* fx, double *ppgk,int64_t ppgk_offset_3_3, int64_t ppgk_offset_3_2, int64_t ppgk_offset_3_1, double *ppgc,int64_t ppgc_offset_3_3, int64_t ppgc_offset_3_2, int64_t ppgc_offset_3_1, double *ppgu,int64_t ppgu_offset_3_3, int64_t ppgu_offset_3_2, int64_t ppgu_offset_3_1,
                 double *ppgv,int64_t ppgv_offset_3_3, int64_t ppgv_offset_3_2, int64_t ppgv_offset_3_1, double edadlat, double dt, int64_t domain_size, int64_t domain_height) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size) * (domain_size) * (domain_height)) return;
        int i = tid / (domain_size) / (domain_height);
        int j = (tid % ((domain_size) * (domain_height))) / (domain_height); 
        int k = tid % (domain_height);
        uout[(i) * uout_offset_3_3 + ( j) * uout_offset_3_2 + ( k) * uout_offset_3_1] =
            uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] +
            dt * (utens[(i) * utens_offset_3_3 + ( j) * utens_offset_3_2 + ( k) * utens_offset_3_1] - ppgu[(i) * ppgu_offset_3_3 + ( j) * ppgu_offset_3_2 + ( k) * ppgu_offset_3_1] * ((2.0) * fx[(j)] / (rho[(i + 1) * rho_offset_3_3 + ( j) * rho_offset_3_2 + ( k) * rho_offset_3_1] + rho[(i) * rho_offset_3_3 + ( j) * rho_offset_3_2 + ( k) * rho_offset_3_1])));


}

__global__ void fastwavesuv_k6(
        double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, double *utens,int64_t utens_offset_3_3, int64_t utens_offset_3_2, int64_t utens_offset_3_1,
                 double *vtens,int64_t vtens_offset_3_3, int64_t vtens_offset_3_2, int64_t vtens_offset_3_1, double *wgtfac,int64_t wgtfac_offset_3_3, int64_t wgtfac_offset_3_2, int64_t wgtfac_offset_3_1, double *ppuv,int64_t ppuv_offset_3_3, int64_t ppuv_offset_3_2, int64_t ppuv_offset_3_1, double *hhl,int64_t hhl_offset_3_3, int64_t hhl_offset_3_2, int64_t hhl_offset_3_1,
                 double *rho,int64_t rho_offset_3_3, int64_t rho_offset_3_2, int64_t rho_offset_3_1, double* fx, double *ppgk,int64_t ppgk_offset_3_3, int64_t ppgk_offset_3_2, int64_t ppgk_offset_3_1, double *ppgc,int64_t ppgc_offset_3_3, int64_t ppgc_offset_3_2, int64_t ppgc_offset_3_1, double *ppgu,int64_t ppgu_offset_3_3, int64_t ppgu_offset_3_2, int64_t ppgu_offset_3_1,
                 double *ppgv,int64_t ppgv_offset_3_3, int64_t ppgv_offset_3_2, int64_t ppgv_offset_3_1, double edadlat, double dt, int64_t domain_size, int64_t domain_height) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size) * (domain_size) * (domain_height)) return;
        int i = tid / (domain_size) / (domain_height);
        int j = (tid % ((domain_size) * (domain_height))) / (domain_height); 
        int k = tid % (domain_height);
        vout[(i) * vout_offset_3_3 + ( j) * vout_offset_3_2 + ( k) * vout_offset_3_1] =
            vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] +
            dt * (vtens[(i) * vtens_offset_3_3 + ( j) * vtens_offset_3_2 + ( k) * vtens_offset_3_1] - ppgv[(i) * ppgv_offset_3_3 + ( j) * ppgv_offset_3_2 + ( k) * ppgv_offset_3_1] * ((2.0) * edadlat / (rho[(i) * rho_offset_3_3 + ( j + 1) * rho_offset_3_2 + ( k) * rho_offset_3_1] + rho[(i) * rho_offset_3_3 + ( j) * rho_offset_3_2 + ( k) * rho_offset_3_1])));


}

void fastwavesuv(
        double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, double *utens,int64_t utens_offset_3_3, int64_t utens_offset_3_2, int64_t utens_offset_3_1,
                 double *vtens,int64_t vtens_offset_3_3, int64_t vtens_offset_3_2, int64_t vtens_offset_3_1, double *wgtfac,int64_t wgtfac_offset_3_3, int64_t wgtfac_offset_3_2, int64_t wgtfac_offset_3_1, double *ppuv,int64_t ppuv_offset_3_3, int64_t ppuv_offset_3_2, int64_t ppuv_offset_3_1, double *hhl,int64_t hhl_offset_3_3, int64_t hhl_offset_3_2, int64_t hhl_offset_3_1,
                 double *rho,int64_t rho_offset_3_3, int64_t rho_offset_3_2, int64_t rho_offset_3_1, double* fx, double *ppgk,int64_t ppgk_offset_3_3, int64_t ppgk_offset_3_2, int64_t ppgk_offset_3_1, double *ppgc,int64_t ppgc_offset_3_3, int64_t ppgc_offset_3_2, int64_t ppgc_offset_3_1, double *ppgu,int64_t ppgu_offset_3_3, int64_t ppgu_offset_3_2, int64_t ppgu_offset_3_1,
                 double *ppgv,int64_t ppgv_offset_3_3, int64_t ppgv_offset_3_2, int64_t ppgv_offset_3_1, double edadlat, double dt) {
    int64_t size;
    int blocksize = 128;
    size = (domain_size + 1) * (domain_size + 1) * (domain_height + 1);
    fastwavesuv_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1, uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1, vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1, utens, utens_offset_3_3,  utens_offset_3_2,  utens_offset_3_1, vtens, vtens_offset_3_3,  vtens_offset_3_2,  vtens_offset_3_1, wgtfac, wgtfac_offset_3_3,  wgtfac_offset_3_2,  wgtfac_offset_3_1, ppuv, ppuv_offset_3_3,  ppuv_offset_3_2,  ppuv_offset_3_1, hhl, hhl_offset_3_3,  hhl_offset_3_2,  hhl_offset_3_1, rho, rho_offset_3_3,  rho_offset_3_2,  rho_offset_3_1,  fx, ppgk, ppgk_offset_3_3,  ppgk_offset_3_2,  ppgk_offset_3_1, ppgc, ppgc_offset_3_3,  ppgc_offset_3_2,  ppgc_offset_3_1, ppgu, ppgu_offset_3_3,  ppgu_offset_3_2,  ppgu_offset_3_1, ppgv, ppgv_offset_3_3,  ppgv_offset_3_2,  ppgv_offset_3_1, edadlat, dt,  domain_size,  domain_height);
    fastwavesuv_k2<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1, uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1, vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1, utens, utens_offset_3_3,  utens_offset_3_2,  utens_offset_3_1, vtens, vtens_offset_3_3,  vtens_offset_3_2,  vtens_offset_3_1, wgtfac, wgtfac_offset_3_3,  wgtfac_offset_3_2,  wgtfac_offset_3_1, ppuv, ppuv_offset_3_3,  ppuv_offset_3_2,  ppuv_offset_3_1, hhl, hhl_offset_3_3,  hhl_offset_3_2,  hhl_offset_3_1, rho, rho_offset_3_3,  rho_offset_3_2,  rho_offset_3_1,  fx, ppgk, ppgk_offset_3_3,  ppgk_offset_3_2,  ppgk_offset_3_1, ppgc, ppgc_offset_3_3,  ppgc_offset_3_2,  ppgc_offset_3_1, ppgu, ppgu_offset_3_3,  ppgu_offset_3_2,  ppgu_offset_3_1, ppgv, ppgv_offset_3_3,  ppgv_offset_3_2,  ppgv_offset_3_1, edadlat, dt,  domain_size,  domain_height);
    size = (domain_size + 1) * (domain_size + 1) * (domain_height);
    fastwavesuv_k3<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1, uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1, vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1, utens, utens_offset_3_3,  utens_offset_3_2,  utens_offset_3_1, vtens, vtens_offset_3_3,  vtens_offset_3_2,  vtens_offset_3_1, wgtfac, wgtfac_offset_3_3,  wgtfac_offset_3_2,  wgtfac_offset_3_1, ppuv, ppuv_offset_3_3,  ppuv_offset_3_2,  ppuv_offset_3_1, hhl, hhl_offset_3_3,  hhl_offset_3_2,  hhl_offset_3_1, rho, rho_offset_3_3,  rho_offset_3_2,  rho_offset_3_1,  fx, ppgk, ppgk_offset_3_3,  ppgk_offset_3_2,  ppgk_offset_3_1, ppgc, ppgc_offset_3_3,  ppgc_offset_3_2,  ppgc_offset_3_1, ppgu, ppgu_offset_3_3,  ppgu_offset_3_2,  ppgu_offset_3_1, ppgv, ppgv_offset_3_3,  ppgv_offset_3_2,  ppgv_offset_3_1, edadlat, dt,  domain_size,  domain_height);
    size = (domain_size) * (domain_size) * (domain_height);
    fastwavesuv_k4<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1, uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1, vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1, utens, utens_offset_3_3,  utens_offset_3_2,  utens_offset_3_1, vtens, vtens_offset_3_3,  vtens_offset_3_2,  vtens_offset_3_1, wgtfac, wgtfac_offset_3_3,  wgtfac_offset_3_2,  wgtfac_offset_3_1, ppuv, ppuv_offset_3_3,  ppuv_offset_3_2,  ppuv_offset_3_1, hhl, hhl_offset_3_3,  hhl_offset_3_2,  hhl_offset_3_1, rho, rho_offset_3_3,  rho_offset_3_2,  rho_offset_3_1,  fx, ppgk, ppgk_offset_3_3,  ppgk_offset_3_2,  ppgk_offset_3_1, ppgc, ppgc_offset_3_3,  ppgc_offset_3_2,  ppgc_offset_3_1, ppgu, ppgu_offset_3_3,  ppgu_offset_3_2,  ppgu_offset_3_1, ppgv, ppgv_offset_3_3,  ppgv_offset_3_2,  ppgv_offset_3_1, edadlat, dt,  domain_size,  domain_height);
    fastwavesuv_k5<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1, uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1, vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1, utens, utens_offset_3_3,  utens_offset_3_2,  utens_offset_3_1, vtens, vtens_offset_3_3,  vtens_offset_3_2,  vtens_offset_3_1, wgtfac, wgtfac_offset_3_3,  wgtfac_offset_3_2,  wgtfac_offset_3_1, ppuv, ppuv_offset_3_3,  ppuv_offset_3_2,  ppuv_offset_3_1, hhl, hhl_offset_3_3,  hhl_offset_3_2,  hhl_offset_3_1, rho, rho_offset_3_3,  rho_offset_3_2,  rho_offset_3_1,  fx, ppgk, ppgk_offset_3_3,  ppgk_offset_3_2,  ppgk_offset_3_1, ppgc, ppgc_offset_3_3,  ppgc_offset_3_2,  ppgc_offset_3_1, ppgu, ppgu_offset_3_3,  ppgu_offset_3_2,  ppgu_offset_3_1, ppgv, ppgv_offset_3_3,  ppgv_offset_3_2,  ppgv_offset_3_1, edadlat, dt,  domain_size,  domain_height);
    fastwavesuv_k6<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1, uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1, vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1, utens, utens_offset_3_3,  utens_offset_3_2,  utens_offset_3_1, vtens, vtens_offset_3_3,  vtens_offset_3_2,  vtens_offset_3_1, wgtfac, wgtfac_offset_3_3,  wgtfac_offset_3_2,  wgtfac_offset_3_1, ppuv, ppuv_offset_3_3,  ppuv_offset_3_2,  ppuv_offset_3_1, hhl, hhl_offset_3_3,  hhl_offset_3_2,  hhl_offset_3_1, rho, rho_offset_3_3,  rho_offset_3_2,  rho_offset_3_1,  fx, ppgk, ppgk_offset_3_3,  ppgk_offset_3_2,  ppgk_offset_3_1, ppgc, ppgc_offset_3_3,  ppgc_offset_3_2,  ppgc_offset_3_1, ppgu, ppgu_offset_3_3,  ppgu_offset_3_2,  ppgu_offset_3_1, ppgv, ppgv_offset_3_3,  ppgv_offset_3_2,  ppgv_offset_3_1, edadlat, dt,  domain_size,  domain_height);
}

__global__ void fastwavesuv_fullfusion_k1(
        double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, double *utens,int64_t utens_offset_3_3, int64_t utens_offset_3_2, int64_t utens_offset_3_1,
                 double *vtens,int64_t vtens_offset_3_3, int64_t vtens_offset_3_2, int64_t vtens_offset_3_1, double *wgtfac,int64_t wgtfac_offset_3_3, int64_t wgtfac_offset_3_2, int64_t wgtfac_offset_3_1, double *ppuv,int64_t ppuv_offset_3_3, int64_t ppuv_offset_3_2, int64_t ppuv_offset_3_1, double *hhl,int64_t hhl_offset_3_3, int64_t hhl_offset_3_2, int64_t hhl_offset_3_1,
                 double *rho,int64_t rho_offset_3_3, int64_t rho_offset_3_2, int64_t rho_offset_3_1, double* fx, double *ppgk,int64_t ppgk_offset_3_3, int64_t ppgk_offset_3_2, int64_t ppgk_offset_3_1, double *ppgc,int64_t ppgc_offset_3_3, int64_t ppgc_offset_3_2, int64_t ppgc_offset_3_1, double *ppgu,int64_t ppgu_offset_3_3, int64_t ppgu_offset_3_2, int64_t ppgu_offset_3_1,
                 double *ppgv,int64_t ppgv_offset_3_3, int64_t ppgv_offset_3_2, int64_t ppgv_offset_3_1, double edadlat, double dt, int64_t domain_size, int64_t domain_height) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size) * (domain_size) * (domain_height)) return;
        int i = tid / (domain_size) / (domain_height);
        int j = (tid % ((domain_size) * (domain_height))) / (domain_height); 
        int k = tid % (domain_height);

        auto _ppgk_ijk = wgtfac[(i) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k) * wgtfac_offset_3_1] * ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1] + ((1.0) - wgtfac[(i) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k) * wgtfac_offset_3_1]) * ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k - 1) * ppuv_offset_3_1];
        auto _ppgk_i1jk = wgtfac[(i + 1) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k) * wgtfac_offset_3_1] * ppuv[(i + 1) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1] + ((1.0) - wgtfac[(i + 1) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k) * wgtfac_offset_3_1]) * ppuv[(i + 1) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k - 1) * ppuv_offset_3_1];
        auto _ppgk_ij1k = wgtfac[(i) * wgtfac_offset_3_3 + ( j + 1) * wgtfac_offset_3_2 + ( k) * wgtfac_offset_3_1] * ppuv[(i) * ppuv_offset_3_3 + ( j + 1) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1] + ((1.0) - wgtfac[(i) * wgtfac_offset_3_3 + ( j + 1) * wgtfac_offset_3_2 + ( k) * wgtfac_offset_3_1]) * ppuv[(i) * ppuv_offset_3_3 + ( j + 1) * ppuv_offset_3_2 + ( k - 1) * ppuv_offset_3_1];
        auto _ppgk_ijk1 = wgtfac[(i) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k + 1) * wgtfac_offset_3_1] * ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k + 1) * ppuv_offset_3_1] + ((1.0) - wgtfac[(i) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k + 1) * wgtfac_offset_3_1]) * ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1];
        auto _ppgk_i1jk1 = wgtfac[(i + 1) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k + 1) * wgtfac_offset_3_1] * ppuv[(i + 1) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k + 1) * ppuv_offset_3_1] + ((1.0) - wgtfac[(i + 1) * wgtfac_offset_3_3 + ( j) * wgtfac_offset_3_2 + ( k + 1) * wgtfac_offset_3_1]) * ppuv[(i + 1) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1];
        auto _ppgk_ij1k1 = wgtfac[(i) * wgtfac_offset_3_3 + ( j + 1) * wgtfac_offset_3_2 + ( k + 1) * wgtfac_offset_3_1] * ppuv[(i) * ppuv_offset_3_3 + ( j + 1) * ppuv_offset_3_2 + ( k + 1) * ppuv_offset_3_1] + ((1.0) - wgtfac[(i) * wgtfac_offset_3_3 + ( j + 1) * wgtfac_offset_3_2 + ( k + 1) * wgtfac_offset_3_1]) * ppuv[(i) * ppuv_offset_3_3 + ( j + 1) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1];

        auto _ppgc_ijk = _ppgk_ijk1 - _ppgk_ijk;
        auto _ppgc_i1jk = _ppgk_i1jk1 - _ppgk_i1jk;
        auto _ppgc_ij1k = _ppgk_ij1k1 - _ppgk_ij1k;

        auto _ppgu = (ppuv[(i + 1) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1] - ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1]) +
                     (_ppgc_i1jk + _ppgc_ijk) * (0.5) *
                         ((hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] + hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]) - (hhl[(i + 1) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] + hhl[(i + 1) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1])) /
                         ((hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] - hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]) + (hhl[(i + 1) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] - hhl[(i + 1) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]));

        auto _ppgv = (ppuv[(i) * ppuv_offset_3_3 + ( j + 1) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1] - ppuv[(i) * ppuv_offset_3_3 + ( j) * ppuv_offset_3_2 + ( k) * ppuv_offset_3_1]) +
                     (_ppgc_ij1k + _ppgc_ijk) * (0.5) *
                         ((hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] + hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]) - (hhl[(i) * hhl_offset_3_3 + ( j + 1) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] + hhl[(i) * hhl_offset_3_3 + ( j + 1) * hhl_offset_3_2 + ( k) * hhl_offset_3_1])) /
                         ((hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] - hhl[(i) * hhl_offset_3_3 + ( j) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]) + (hhl[(i) * hhl_offset_3_3 + ( j + 1) * hhl_offset_3_2 + ( k + 1) * hhl_offset_3_1] - hhl[(i) * hhl_offset_3_3 + ( j + 1) * hhl_offset_3_2 + ( k) * hhl_offset_3_1]));

        uout[(i) * uout_offset_3_3 + ( j) * uout_offset_3_2 + ( k) * uout_offset_3_1] = uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] +
                        dt * (utens[(i) * utens_offset_3_3 + ( j) * utens_offset_3_2 + ( k) * utens_offset_3_1] - _ppgu * ((2.0) * fx[(j)] / (rho[(i + 1) * rho_offset_3_3 + ( j) * rho_offset_3_2 + ( k) * rho_offset_3_1] + rho[(i) * rho_offset_3_3 + ( j) * rho_offset_3_2 + ( k) * rho_offset_3_1])));

        vout[(i) * vout_offset_3_3 + ( j) * vout_offset_3_2 + ( k) * vout_offset_3_1] = vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] + dt * (vtens[(i) * vtens_offset_3_3 + ( j) * vtens_offset_3_2 + ( k) * vtens_offset_3_1] -
                                             _ppgv * ((2.0) * edadlat / (rho[(i) * rho_offset_3_3 + ( j + 1) * rho_offset_3_2 + ( k) * rho_offset_3_1] + rho[(i) * rho_offset_3_3 + ( j) * rho_offset_3_2 + ( k) * rho_offset_3_1])));
}


void fastwavesuv_fullfusion(
        double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1, double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1, double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1, double *utens,int64_t utens_offset_3_3, int64_t utens_offset_3_2, int64_t utens_offset_3_1,
                 double *vtens,int64_t vtens_offset_3_3, int64_t vtens_offset_3_2, int64_t vtens_offset_3_1, double *wgtfac,int64_t wgtfac_offset_3_3, int64_t wgtfac_offset_3_2, int64_t wgtfac_offset_3_1, double *ppuv,int64_t ppuv_offset_3_3, int64_t ppuv_offset_3_2, int64_t ppuv_offset_3_1, double *hhl,int64_t hhl_offset_3_3, int64_t hhl_offset_3_2, int64_t hhl_offset_3_1,
                 double *rho,int64_t rho_offset_3_3, int64_t rho_offset_3_2, int64_t rho_offset_3_1, double* fx, double *ppgk,int64_t ppgk_offset_3_3, int64_t ppgk_offset_3_2, int64_t ppgk_offset_3_1, double *ppgc,int64_t ppgc_offset_3_3, int64_t ppgc_offset_3_2, int64_t ppgc_offset_3_1, double *ppgu,int64_t ppgu_offset_3_3, int64_t ppgu_offset_3_2, int64_t ppgu_offset_3_1,
                 double *ppgv,int64_t ppgv_offset_3_3, int64_t ppgv_offset_3_2, int64_t ppgv_offset_3_1, double edadlat, double dt) {

    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 128;
    fastwavesuv_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1, uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1, vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1, utens, utens_offset_3_3,  utens_offset_3_2,  utens_offset_3_1, vtens, vtens_offset_3_3,  vtens_offset_3_2,  vtens_offset_3_1, wgtfac, wgtfac_offset_3_3,  wgtfac_offset_3_2,  wgtfac_offset_3_1, ppuv, ppuv_offset_3_3,  ppuv_offset_3_2,  ppuv_offset_3_1, hhl, hhl_offset_3_3,  hhl_offset_3_2,  hhl_offset_3_1, rho, rho_offset_3_3,  rho_offset_3_2,  rho_offset_3_1,  fx, ppgk, ppgk_offset_3_3,  ppgk_offset_3_2,  ppgk_offset_3_1, ppgc, ppgc_offset_3_3,  ppgc_offset_3_2,  ppgc_offset_3_1, ppgu, ppgu_offset_3_3,  ppgu_offset_3_2,  ppgu_offset_3_1, ppgv, ppgv_offset_3_3,  ppgv_offset_3_2,  ppgv_offset_3_1, edadlat, dt,  domain_size,  domain_height);
}


#endif  // FASTWAVESUV_H
