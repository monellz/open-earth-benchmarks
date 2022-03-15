#ifndef P_GRAD_C_H
#define P_GRAD_C_H


__global__ void p_grad_c_fullfusion_k1(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1,  double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1,  double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1,  double *rdxc,int64_t rdxc_offset_3_3, int64_t rdxc_offset_3_2, int64_t rdxc_offset_3_1,
               double *rdyc,int64_t rdyc_offset_3_3, int64_t rdyc_offset_3_2, int64_t rdyc_offset_3_1,  double *delpc,int64_t delpc_offset_3_3, int64_t delpc_offset_3_2, int64_t delpc_offset_3_1,  double *gz,int64_t gz_offset_3_3, int64_t gz_offset_3_2, int64_t gz_offset_3_1,  double *pkc,int64_t pkc_offset_3_3, int64_t pkc_offset_3_2, int64_t pkc_offset_3_1, double *wk,int64_t wk_offset_3_3, int64_t wk_offset_3_2, int64_t wk_offset_3_1,
               double dt2, int64_t domain_size, int64_t domain_height) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size) * (domain_size) * (domain_height)) return;
        int k = tid / (domain_size) / (domain_size);
        int i = (tid % ((domain_size) * (domain_size))) / (domain_size);
        int j = tid % (domain_size);
  //for (int64_t k = 0; k < domain_height; ++k) {
  //  for (int64_t i = 0; i < domain_size; ++i) {
  //    for (int64_t j = 0; j < domain_size; ++j) {
        auto _wk_ijk = delpc[(i) * delpc_offset_3_3 + ( j) * delpc_offset_3_2 + ( k) * delpc_offset_3_1];
        auto _wk_im1jk = delpc[(i - 1) * delpc_offset_3_3 + ( j) * delpc_offset_3_2 + ( k) * delpc_offset_3_1];
        auto _wk_ijm1k = delpc[(i) * delpc_offset_3_3 + ( j - 1) * delpc_offset_3_2 + ( k) * delpc_offset_3_1];
        uout[(i) * uout_offset_3_3 + ( j) * uout_offset_3_2 + ( k) * uout_offset_3_1] =
            (uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] + (((dt2 * rdxc[(i) * rdxc_offset_3_3 + ( j) * rdxc_offset_3_2 + ( k) * rdxc_offset_3_1]) / (_wk_im1jk + _wk_ijk)) *
                             (((gz[(i - 1) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1] - gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k) * gz_offset_3_1]) * (pkc[(i) * pkc_offset_3_3 + ( j) * pkc_offset_3_2 + ( k + 1) * pkc_offset_3_1] - pkc[(i - 1) * pkc_offset_3_3 + ( j) * pkc_offset_3_2 + ( k) * pkc_offset_3_1])) +
                              ((gz[(i - 1) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k) * gz_offset_3_1] - gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1]) * (pkc[(i - 1) * pkc_offset_3_3 + ( j) * pkc_offset_3_2 + ( k + 1) * pkc_offset_3_1] - pkc[(i) * pkc_offset_3_3 + ( j) * pkc_offset_3_2 + ( k) * pkc_offset_3_1])))));
        vout[(i) * vout_offset_3_3 + ( j) * vout_offset_3_2 + ( k) * vout_offset_3_1] =
            (vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] + (((dt2 * rdyc[(i) * rdyc_offset_3_3 + ( j) * rdyc_offset_3_2 + ( k) * rdyc_offset_3_1]) / (_wk_ijm1k + _wk_ijk)) *
                             (((gz[(i) * gz_offset_3_3 + ( j - 1) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1] - gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k) * gz_offset_3_1]) * (pkc[(i) * pkc_offset_3_3 + ( j) * pkc_offset_3_2 + ( k + 1) * pkc_offset_3_1] - pkc[(i) * pkc_offset_3_3 + ( j - 1) * pkc_offset_3_2 + ( k) * pkc_offset_3_1])) +
                              ((gz[(i) * gz_offset_3_3 + ( j - 1) * gz_offset_3_2 + ( k) * gz_offset_3_1] - gz[(i) * gz_offset_3_3 + ( j) * gz_offset_3_2 + ( k + 1) * gz_offset_3_1]) * (pkc[(i) * pkc_offset_3_3 + ( j - 1) * pkc_offset_3_2 + ( k + 1) * pkc_offset_3_1] - pkc[(i) * pkc_offset_3_3 + ( j) * pkc_offset_3_2 + ( k) * pkc_offset_3_1])))));
  //    }
  //  }
  //}
}


void p_grad_c_fullfusion(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1,  double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1,  double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1,  double *rdxc,int64_t rdxc_offset_3_3, int64_t rdxc_offset_3_2, int64_t rdxc_offset_3_1,
               double *rdyc,int64_t rdyc_offset_3_3, int64_t rdyc_offset_3_2, int64_t rdyc_offset_3_1,  double *delpc,int64_t delpc_offset_3_3, int64_t delpc_offset_3_2, int64_t delpc_offset_3_1,  double *gz,int64_t gz_offset_3_3, int64_t gz_offset_3_2, int64_t gz_offset_3_1,  double *pkc,int64_t pkc_offset_3_3, int64_t pkc_offset_3_2, int64_t pkc_offset_3_1, double *wk,int64_t wk_offset_3_3, int64_t wk_offset_3_2, int64_t wk_offset_3_1,
               double dt2) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 64;
    p_grad_c_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (
         uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1,  uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1,  vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1,  rdxc, rdxc_offset_3_3,  rdxc_offset_3_2,  rdxc_offset_3_1,
               rdyc, rdyc_offset_3_3,  rdyc_offset_3_2,  rdyc_offset_3_1,  delpc, delpc_offset_3_3,  delpc_offset_3_2,  delpc_offset_3_1,  gz, gz_offset_3_3,  gz_offset_3_2,  gz_offset_3_1,  pkc, pkc_offset_3_3,  pkc_offset_3_2,  pkc_offset_3_1, wk, wk_offset_3_3,  wk_offset_3_2,  wk_offset_3_1,
                dt2, domain_size, domain_height
            );
}


void p_grad_c_fullfusion2(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1,  double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1,  double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1,  double *rdxc,int64_t rdxc_offset_3_3, int64_t rdxc_offset_3_2, int64_t rdxc_offset_3_1,
               double *rdyc,int64_t rdyc_offset_3_3, int64_t rdyc_offset_3_2, int64_t rdyc_offset_3_1,  double *delpc,int64_t delpc_offset_3_3, int64_t delpc_offset_3_2, int64_t delpc_offset_3_1,  double *gz,int64_t gz_offset_3_3, int64_t gz_offset_3_2, int64_t gz_offset_3_1,  double *pkc,int64_t pkc_offset_3_3, int64_t pkc_offset_3_2, int64_t pkc_offset_3_1, double *wk,int64_t wk_offset_3_3, int64_t wk_offset_3_2, int64_t wk_offset_3_1,
               double dt2) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 512;
    p_grad_c_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (
         uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1,  uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1,  vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1,  rdxc, rdxc_offset_3_3,  rdxc_offset_3_2,  rdxc_offset_3_1,
               rdyc, rdyc_offset_3_3,  rdyc_offset_3_2,  rdyc_offset_3_1,  delpc, delpc_offset_3_3,  delpc_offset_3_2,  delpc_offset_3_1,  gz, gz_offset_3_3,  gz_offset_3_2,  gz_offset_3_1,  pkc, pkc_offset_3_3,  pkc_offset_3_2,  pkc_offset_3_1, wk, wk_offset_3_3,  wk_offset_3_2,  wk_offset_3_1,
                dt2, domain_size, domain_height
            );
}
#endif  // P_GRAD_C_H
