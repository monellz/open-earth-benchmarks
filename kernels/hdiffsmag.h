#ifndef HDIFFSMAG_H
#define HDIFFSMAG_H

__global__ void hdiffsmag_fullfusion_k1(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1,  double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1,  double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1,  double *mask,int64_t mask_offset_3_3, int64_t mask_offset_3_2, int64_t mask_offset_3_1,
                double * crlavo,  double * crlavu,  double * crlato,  double * crlatu,
                double * acrlat0, double *T_sqr_s,int64_t T_sqr_s_offset_3_3, int64_t T_sqr_s_offset_3_2, int64_t T_sqr_s_offset_3_1, double *S_sqr_uv,int64_t S_sqr_uv_offset_3_3, int64_t S_sqr_uv_offset_3_2, int64_t S_sqr_uv_offset_3_1,  ElementType eddlat,
                ElementType eddlon,  ElementType tau_smag,  ElementType weight_smag, int64_t domain_size, int64_t domain_height) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size) * (domain_size) * (domain_height)) return;
        int i = tid / (domain_size) / (domain_height);
        int j = (tid % ((domain_size) * (domain_height))) / (domain_height);
        int k = tid % (domain_height);
  //for (int64_t i = 0; i < domain_size; i++) {
  //  for (int64_t j = 0; j < domain_size; j++) {
  //    for (int64_t k = 0; k < domain_height; k++) {
        auto T_s_ijk = (vin[(i) * vin_offset_3_3 + ( j - 1) * vin_offset_3_2 + ( k) * vin_offset_3_1] - vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1]) * eddlat * EARTH_RADIUS_RECIP - (uin[(i - 1) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] - uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1]) * acrlat0[j] * eddlon;
        auto T_sqr_s_ijk = T_s_ijk * T_s_ijk;
        auto T_s_i1jk = (vin[(i + 1) * vin_offset_3_3 + ( j - 1) * vin_offset_3_2 + ( k) * vin_offset_3_1] - vin[(i + 1) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1]) * eddlat * EARTH_RADIUS_RECIP - (uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] - uin[(i + 1) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1]) * acrlat0[j] * eddlon;
        auto T_sqr_s_i1jk = T_s_i1jk * T_s_i1jk;
        auto T_s_ij1k = (vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] - vin[(i) * vin_offset_3_3 + ( j + 1) * vin_offset_3_2 + ( k) * vin_offset_3_1]) * eddlat * EARTH_RADIUS_RECIP - (uin[(i - 1) * uin_offset_3_3 + ( j + 1) * uin_offset_3_2 + ( k) * uin_offset_3_1] - uin[(i) * uin_offset_3_3 + ( j + 1) * uin_offset_3_2 + ( k) * uin_offset_3_1]) * acrlat0[j + 1] * eddlon;
        auto T_sqr_s_ij1k = T_s_ij1k * T_s_ij1k;


        auto S_uv_ijk = (uin[(i) * uin_offset_3_3 + ( j + 1) * uin_offset_3_2 + ( k) * uin_offset_3_1] - uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1]) * eddlat * EARTH_RADIUS_RECIP + (vin[(i + 1) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] - vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1]) * acrlat0[j] * eddlon;
        auto S_sqr_uv_ijk = S_uv_ijk * S_uv_ijk;
        auto S_uv_im1jk = (uin[(i - 1) * uin_offset_3_3 + ( j + 1) * uin_offset_3_2 + ( k) * uin_offset_3_1] - uin[(i - 1) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1]) * eddlat * EARTH_RADIUS_RECIP + (vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] - vin[(i - 1) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1]) * acrlat0[j] * eddlon;
        auto S_sqr_uv_im1jk = S_uv_im1jk * S_uv_im1jk;
        auto S_uv_ijm1k = (uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] - uin[(i) * uin_offset_3_3 + ( j - 1) * uin_offset_3_2 + ( k) * uin_offset_3_1]) * eddlat * EARTH_RADIUS_RECIP + (vin[(i + 1) * vin_offset_3_3 + ( j - 1) * vin_offset_3_2 + ( k) * vin_offset_3_1] - vin[(i) * vin_offset_3_3 + ( j - 1) * vin_offset_3_2 + ( k) * vin_offset_3_1]) * acrlat0[j - 1] * eddlon;
        auto S_sqr_uv_ijm1k = S_uv_ijm1k * S_uv_ijm1k;


         ElementType hdweight = weight_smag * mask[(i) * mask_offset_3_3 + ( j) * mask_offset_3_2 + ( k) * mask_offset_3_1];

        // I direction
        // valid on [1:I-1,1:J-1,O:K]
        ElementType smag_u = tau_smag * std::sqrt(ElementType(0.5) * (T_sqr_s_i1jk + T_sqr_s_ijk) +
                                                  ElementType(0.5) * (S_sqr_uv_ijm1k + S_sqr_uv_ijk)) -
                             hdweight;
        smag_u = fmin(ElementType(0.5), fmax(ElementType(0.0), smag_u));

        // valid on [1:I-1,1:J-1,0:K]
         ElementType lapu = uin[(i + 1) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] + uin[(i - 1) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] - ElementType(2.0) * uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] +
                                 crlato[j] * (uin[(i) * uin_offset_3_3 + ( j + 1) * uin_offset_3_2 + ( k) * uin_offset_3_1] - uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1]) +
                                 crlatu[j] * (uin[(i) * uin_offset_3_3 + ( j - 1) * uin_offset_3_2 + ( k) * uin_offset_3_1] - uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1]);

        // valid on [1:I-1,1:J-1,0:K]
        uout[(i) * uout_offset_3_3 + ( j) * uout_offset_3_2 + ( k) * uout_offset_3_1] = uin[(i) * uin_offset_3_3 + ( j) * uin_offset_3_2 + ( k) * uin_offset_3_1] + smag_u * lapu;

        // J direction
        // valid on [1:I-1,1:J-1,0:K]
        ElementType smag_v = tau_smag * std::sqrt(ElementType(0.5) * (T_sqr_s_ij1k + T_sqr_s_ijk) +
                                                  ElementType(0.5) * (S_sqr_uv_im1jk + S_sqr_uv_ijk)) -
                             hdweight;
        smag_v = fmin(ElementType(0.5), fmax(ElementType(0.0), smag_v));

        // valid on [1:I-1,1:J-1,0:K]
         ElementType lapv = vin[(i + 1) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] + vin[(i - 1) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] - ElementType(2.0) * vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] +
                                 crlavo[j] * (vin[(i) * vin_offset_3_3 + ( j + 1) * vin_offset_3_2 + ( k) * vin_offset_3_1] - vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1]) +
                                 crlavu[j] * (vin[(i) * vin_offset_3_3 + ( j - 1) * vin_offset_3_2 + ( k) * vin_offset_3_1] - vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1]);

        // valid on [1:I-1,1:J-1,0:K]
        vout[(i) * vout_offset_3_3 + ( j) * vout_offset_3_2 + ( k) * vout_offset_3_1] = vin[(i) * vin_offset_3_3 + ( j) * vin_offset_3_2 + ( k) * vin_offset_3_1] + smag_v * lapv;
//      }
//    }
//  }
}


void hdiffsmag_fullfusion(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1,  double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1,  double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1,  double *mask,int64_t mask_offset_3_3, int64_t mask_offset_3_2, int64_t mask_offset_3_1,
                double * crlavo,  double * crlavu,  double * crlato,  double * crlatu,
                double * acrlat0, double *T_sqr_s,int64_t T_sqr_s_offset_3_3, int64_t T_sqr_s_offset_3_2, int64_t T_sqr_s_offset_3_1, double *S_sqr_uv,int64_t S_sqr_uv_offset_3_3, int64_t S_sqr_uv_offset_3_2, int64_t S_sqr_uv_offset_3_1,  ElementType eddlat,
                ElementType eddlon,  ElementType tau_smag,  ElementType weight_smag) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 64;
    hdiffsmag_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (
         uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1,  uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1,  vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1,  mask, mask_offset_3_3,  mask_offset_3_2,  mask_offset_3_1,
                 crlavo,   crlavu,   crlato,   crlatu,
                 acrlat0, T_sqr_s, T_sqr_s_offset_3_3,  T_sqr_s_offset_3_2,  T_sqr_s_offset_3_1, S_sqr_uv, S_sqr_uv_offset_3_3,  S_sqr_uv_offset_3_2,  S_sqr_uv_offset_3_1,   eddlat,
                 eddlon,   tau_smag,   weight_smag,  domain_size,  domain_height
            );
}

void hdiffsmag_fullfusion2(double *uout,int64_t uout_offset_3_3, int64_t uout_offset_3_2, int64_t uout_offset_3_1, double *vout,int64_t vout_offset_3_3, int64_t vout_offset_3_2, int64_t vout_offset_3_1,  double *uin,int64_t uin_offset_3_3, int64_t uin_offset_3_2, int64_t uin_offset_3_1,  double *vin,int64_t vin_offset_3_3, int64_t vin_offset_3_2, int64_t vin_offset_3_1,  double *mask,int64_t mask_offset_3_3, int64_t mask_offset_3_2, int64_t mask_offset_3_1,
                double * crlavo,  double * crlavu,  double * crlato,  double * crlatu,
                double * acrlat0, double *T_sqr_s,int64_t T_sqr_s_offset_3_3, int64_t T_sqr_s_offset_3_2, int64_t T_sqr_s_offset_3_1, double *S_sqr_uv,int64_t S_sqr_uv_offset_3_3, int64_t S_sqr_uv_offset_3_2, int64_t S_sqr_uv_offset_3_1,  ElementType eddlat,
                ElementType eddlon,  ElementType tau_smag,  ElementType weight_smag) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 512;
    hdiffsmag_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (
         uout, uout_offset_3_3,  uout_offset_3_2,  uout_offset_3_1, vout, vout_offset_3_3,  vout_offset_3_2,  vout_offset_3_1,  uin, uin_offset_3_3,  uin_offset_3_2,  uin_offset_3_1,  vin, vin_offset_3_3,  vin_offset_3_2,  vin_offset_3_1,  mask, mask_offset_3_3,  mask_offset_3_2,  mask_offset_3_1,
                 crlavo,   crlavu,   crlato,   crlatu,
                 acrlat0, T_sqr_s, T_sqr_s_offset_3_3,  T_sqr_s_offset_3_2,  T_sqr_s_offset_3_1, S_sqr_uv, S_sqr_uv_offset_3_3,  S_sqr_uv_offset_3_2,  S_sqr_uv_offset_3_1,   eddlat,
                 eddlon,   tau_smag,   weight_smag,  domain_size,  domain_height
            );
}


#endif  // HDIFFSMAG_H
