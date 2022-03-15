#ifndef NH_P_GRAD_H
#define NH_P_GRAD_H

__global__ void nh_p_grad_fullfusion_k1(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1,  double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1,  double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1,  double *rdx,int64_t rdx_offset_3_3, int64_t rdx_offset_3_2, int64_t rdx_offset_3_1,
                double *rdy,int64_t rdy_offset_3_3, int64_t rdy_offset_3_2, int64_t rdy_offset_3_1,  double *gz,int64_t gz_offset_3_3, int64_t gz_offset_3_2, int64_t gz_offset_3_1,  double *pp,int64_t pp_offset_3_3, int64_t pp_offset_3_2, int64_t pp_offset_3_1,  double *pk3,int64_t pk3_offset_3_3, int64_t pk3_offset_3_2, int64_t pk3_offset_3_1,
                double *wk1,int64_t wk1_offset_3_3, int64_t wk1_offset_3_2, int64_t wk1_offset_3_1, double *wk,int64_t wk_offset_3_3, int64_t wk_offset_3_2, int64_t wk_offset_3_1, double *du,int64_t du_offset_3_3, int64_t du_offset_3_2, int64_t du_offset_3_1, double *dv,int64_t dv_offset_3_3, int64_t dv_offset_3_2, int64_t dv_offset_3_1,  ElementType dt, int64_t domain_size, int64_t domain_height) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size) * (domain_size) * (domain_height)) return;
        int k = tid / (domain_size) / (domain_size);
        int i = (tid % ((domain_size) * (domain_size))) / (domain_size);
        int j = tid % (domain_size);
  //for (int64_t k = 0; k < domain_height; ++k) {
  //  for (int64_t i = 0; i < domain_size; ++i) {
  //    for (int64_t j = 0; j < domain_size; ++j) {
        auto wk_ijk = (pk3[(i) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k + 1) * pk3_offset_3_1] - pk3[(i) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k) * pk3_offset_3_1]);
        auto wk_i1jk = (pk3[(i + 1) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k + 1) * pk3_offset_3_1] - pk3[(i + 1) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k) * pk3_offset_3_1]);
        auto wk_ij1k = (pk3[(i) * pk3_offset_3_3 + ( j + 1) * pk3_offset_3_2 + ( k + 1) * pk3_offset_3_1] - pk3[(i) * pk3_offset_3_3 + ( j + 1) * pk3_offset_3_2 + ( k) * pk3_offset_3_1]);
        auto _du = ((dt / (wk_ijk + wk_i1jk)) *
                       (((gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1] - gz[(i + 1) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k) * gz_offset_3_1]) * (pk3[(i + 1) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k + 1) * pk3_offset_3_1] - pk3[(i) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k) * pk3_offset_3_1])) +
                        ((gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k) * gz_offset_3_1] - gz[(i + 1) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1]) * (pk3[(i) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k + 1) * pk3_offset_3_1] - pk3[(i + 1) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k) * pk3_offset_3_1]))));
        uout[(i) * uout_offset_3_3 + ( j) * uout_offset_3_2 + ( k) * uout_offset_3_1] = (((uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] + _du) +
                          ((dt / (wk1[(i) * wk1_offset_3_3 + ( j) * wk1_offset_3_2 + ( k) * wk1_offset_3_1] + wk1[(i + 1) * wk1_offset_3_3 + ( j) * wk1_offset_3_2 + ( k) * wk1_offset_3_1])) *
                           (((gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1] - gz[(i + 1) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k) * gz_offset_3_1]) * (pp[(i + 1) * pp_offset_3_3 + ( j) * pp_offset_3_2 + ( k + 1) * pp_offset_3_1] - pp[(i) * pp_offset_3_3 + ( j) * pp_offset_3_2 + ( k) * pp_offset_3_1])) +
                            ((gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k) * gz_offset_3_1] - gz[(i + 1) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1]) * (pp[(i) * pp_offset_3_3 + ( j) * pp_offset_3_2 + ( k + 1) * pp_offset_3_1] - pp[(i + 1) * pp_offset_3_3 + ( j) * pp_offset_3_2 + ( k) * pp_offset_3_1]))))) *
                         rdx[(i) * rdx_offset_3_3 + ( j) * rdx_offset_3_2 + ( k) * rdx_offset_3_1]);

        auto _dv = ((dt / (wk_ijk + wk_ij1k)) *
                       (((gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1] - gz[(i) * gz_offset_3_3 + ( j + 1) * gz_offset_3_2 + ( k) * gz_offset_3_1]) * (pk3[(i) * pk3_offset_3_3 + ( j + 1) * pk3_offset_3_2 + ( k + 1) * pk3_offset_3_1] - pk3[(i) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k) * pk3_offset_3_1])) +
                        ((gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k) * gz_offset_3_1] - gz[(i) * gz_offset_3_3 + ( j + 1) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1]) * (pk3[(i) * pk3_offset_3_3 + ( j) * pk3_offset_3_2 + ( k + 1) * pk3_offset_3_1] - pk3[(i) * pk3_offset_3_3 + ( j + 1) * pk3_offset_3_2 + ( k) * pk3_offset_3_1]))));
        vout[(i) * vout_offset_3_3 + ( j) * vout_offset_3_2 + ( k) * vout_offset_3_1] = (((vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] + _dv) +
                          ((dt / (wk1[(i) * wk1_offset_3_3 + ( j) * wk1_offset_3_2 + ( k) * wk1_offset_3_1] + wk1[(i) * wk1_offset_3_3 + ( j + 1) * wk1_offset_3_2 + ( k) * wk1_offset_3_1])) *
                           (((gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1] - gz[(i) * gz_offset_3_3 + ( j + 1) * gz_offset_3_2 + ( k) * gz_offset_3_1]) * (pp[(i) * pp_offset_3_3 + ( j + 1) * pp_offset_3_2 + ( k + 1) * pp_offset_3_1] - pp[(i) * pp_offset_3_3 + ( j) * pp_offset_3_2 + ( k) * pp_offset_3_1])) +
                            ((gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k) * gz_offset_3_1] - gz[(i) * gz_offset_3_3 + ( j + 1) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1]) * (pp[(i) * pp_offset_3_3 + ( j) * pp_offset_3_2 + ( k + 1) * pp_offset_3_1] - pp[(i) * pp_offset_3_3 + ( j + 1) * pp_offset_3_2 + ( k) * pp_offset_3_1]))))) *
                         rdy[(i) * rdy_offset_3_3 + ( j) * rdy_offset_3_2 + ( k) * rdy_offset_3_1]);
  //    }
  //  }
  //}
}


void nh_p_grad_fullfusion(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1,  double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1,  double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1,  double *rdx,int64_t rdx_offset_3_3, int64_t rdx_offset_3_2, int64_t rdx_offset_3_1,
                double *rdy,int64_t rdy_offset_3_3, int64_t rdy_offset_3_2, int64_t rdy_offset_3_1,  double *gz,int64_t gz_offset_3_3, int64_t gz_offset_3_2, int64_t gz_offset_3_1,  double *pp,int64_t pp_offset_3_3, int64_t pp_offset_3_2, int64_t pp_offset_3_1,  double *pk3,int64_t pk3_offset_3_3, int64_t pk3_offset_3_2, int64_t pk3_offset_3_1,
                double *wk1,int64_t wk1_offset_3_3, int64_t wk1_offset_3_2, int64_t wk1_offset_3_1, double *wk,int64_t wk_offset_3_3, int64_t wk_offset_3_2, int64_t wk_offset_3_1, double *du,int64_t du_offset_3_3, int64_t du_offset_3_2, int64_t du_offset_3_1, double *dv,int64_t dv_offset_3_3, int64_t dv_offset_3_2, int64_t dv_offset_3_1,  ElementType dt) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 64;
    nh_p_grad_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1,  uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1,  vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1,  rdx, rdx_offset_3_3,  rdx_offset_3_2,  rdx_offset_3_1,
                rdy, rdy_offset_3_3,  rdy_offset_3_2,  rdy_offset_3_1,  gz, gz_offset_3_3,  gz_offset_3_2,  gz_offset_3_1,  pp, pp_offset_3_3,  pp_offset_3_2,  pp_offset_3_1,  pk3, pk3_offset_3_3,  pk3_offset_3_2,  pk3_offset_3_1,
                wk1, wk1_offset_3_3,  wk1_offset_3_2,  wk1_offset_3_1, wk, wk_offset_3_3,  wk_offset_3_2,  wk_offset_3_1, du, du_offset_3_3,  du_offset_3_2,  du_offset_3_1, dv, dv_offset_3_3,  dv_offset_3_2,  dv_offset_3_1,   dt, domain_size, domain_height);
}

void nh_p_grad_fullfusion2(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1,  double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1,  double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1,  double *rdx,int64_t rdx_offset_3_3, int64_t rdx_offset_3_2, int64_t rdx_offset_3_1,
                double *rdy,int64_t rdy_offset_3_3, int64_t rdy_offset_3_2, int64_t rdy_offset_3_1,  double *gz,int64_t gz_offset_3_3, int64_t gz_offset_3_2, int64_t gz_offset_3_1,  double *pp,int64_t pp_offset_3_3, int64_t pp_offset_3_2, int64_t pp_offset_3_1,  double *pk3,int64_t pk3_offset_3_3, int64_t pk3_offset_3_2, int64_t pk3_offset_3_1,
                double *wk1,int64_t wk1_offset_3_3, int64_t wk1_offset_3_2, int64_t wk1_offset_3_1, double *wk,int64_t wk_offset_3_3, int64_t wk_offset_3_2, int64_t wk_offset_3_1, double *du,int64_t du_offset_3_3, int64_t du_offset_3_2, int64_t du_offset_3_1, double *dv,int64_t dv_offset_3_3, int64_t dv_offset_3_2, int64_t dv_offset_3_1,  ElementType dt) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 512;
    nh_p_grad_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1,  uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1,  vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1,  rdx, rdx_offset_3_3,  rdx_offset_3_2,  rdx_offset_3_1,
                rdy, rdy_offset_3_3,  rdy_offset_3_2,  rdy_offset_3_1,  gz, gz_offset_3_3,  gz_offset_3_2,  gz_offset_3_1,  pp, pp_offset_3_3,  pp_offset_3_2,  pp_offset_3_1,  pk3, pk3_offset_3_3,  pk3_offset_3_2,  pk3_offset_3_1,
                wk1, wk1_offset_3_3,  wk1_offset_3_2,  wk1_offset_3_1, wk, wk_offset_3_3,  wk_offset_3_2,  wk_offset_3_1, du, du_offset_3_3,  du_offset_3_2,  du_offset_3_1, dv, dv_offset_3_3,  dv_offset_3_2,  dv_offset_3_1,   dt, domain_size, domain_height);
}


#endif  // NH_P_GRAD_H
