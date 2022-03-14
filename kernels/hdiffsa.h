#ifndef HDIFFSA_H
#define HDIFFSA_H

__global__ void hdiffsa_fullfusion_k1(double *out,int64_t out_offset_3_3, int64_t out_offset_3_2, int64_t out_offset_3_1,  double *in,int64_t in_offset_3_3, int64_t in_offset_3_2, int64_t in_offset_3_1,  double *mask,int64_t mask_offset_3_3, int64_t mask_offset_3_2, int64_t mask_offset_3_1,  double* crlato,
              double* crlatu, double *lap,int64_t lap_offset_3_3, int64_t lap_offset_3_2, int64_t lap_offset_3_1, double *flx,int64_t flx_offset_3_3, int64_t flx_offset_3_2, int64_t flx_offset_3_1, double *fly,int64_t fly_offset_3_3, int64_t fly_offset_3_2, int64_t fly_offset_3_1, int64_t domain_size, int64_t domain_height) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size + 1) * (domain_size + 1) * (domain_height)) return;
        int k = tid / (domain_size) / (domain_size);
        int i = (tid % ((domain_size) * (domain_size))) / (domain_size);
        int j = tid % (domain_size);
  // for (int64_t k = 0; k < domain_height; ++k) {
  //   for (int64_t i = -1; i < domain_size + 1; ++i) {
  //     for (int64_t j = -1; j < domain_size + 1; ++j) {
        auto _lap_ijk = in[(i - 1) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1] + in[(i + 1) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1] - ElementType(2.0) * in[(i) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1] + crlato[j] * (in[(i) * in_offset_3_3 + ( j + 1) * in_offset_3_2 + ( k) * in_offset_3_1] - in[(i) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1]) + crlatu[j] * (in[(i) * in_offset_3_3 + ( j - 1) * in_offset_3_2 + ( k) * in_offset_3_1] - in[(i) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1]);
        auto _lap_i1jk = in[(i) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1] + in[(i + 2) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1] - ElementType(2.0) * in[(i + 1) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1] + crlato[j] * (in[(i + 1) * in_offset_3_3 + ( j + 1) * in_offset_3_2 + ( k) * in_offset_3_1] - in[(i + 1) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1]) + crlatu[j] * (in[(i + 1) * in_offset_3_3 + ( j - 1) * in_offset_3_2 + ( k) * in_offset_3_1] - in[(i + 1) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1]);
        auto _lap_ij1k = in[(i - 1) * in_offset_3_3 + ( j + 1) * in_offset_3_2 + ( k) * in_offset_3_1] + in[(i + 1) * in_offset_3_3 + ( j + 1) * in_offset_3_2 + ( k) * in_offset_3_1] - ElementType(2.0) * in[(i) * in_offset_3_3 + ( j + 1) * in_offset_3_2 + ( k) * in_offset_3_1] + crlato[j + 1] * (in[(i) * in_offset_3_3 + ( j + 2) * in_offset_3_2 + ( k) * in_offset_3_1] - in[(i) * in_offset_3_3 + ( j + 1) * in_offset_3_2 + ( k) * in_offset_3_1]) + crlatu[j + 1] * (in[(i) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1] - in[(i) * in_offset_3_3 + ( j + 1) * in_offset_3_2 + ( k) * in_offset_3_1]);


        flx[(i) * flx_offset_3_3 + ( j) * flx_offset_3_2 + ( k) * flx_offset_3_1] = _lap_i1jk - _lap_ijk;
        if (flx[(i) * flx_offset_3_3 + ( j) * flx_offset_3_2 + ( k) * flx_offset_3_1] * (in[(i + 1) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1] - in[(i) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1]) > 0) {
          flx[(i) * flx_offset_3_3 + ( j) * flx_offset_3_2 + ( k) * flx_offset_3_1] = 0;
        }
        fly[(i) * fly_offset_3_3 + ( j) * fly_offset_3_2 + ( k) * fly_offset_3_1] = crlato[j] * (_lap_ij1k - _lap_ijk);
        if (fly[(i) * fly_offset_3_3 + ( j) * fly_offset_3_2 + ( k) * fly_offset_3_1] * (in[(i) * in_offset_3_3 + ( j + 1) * in_offset_3_2 + ( k) * in_offset_3_1] - in[(i) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1]) > 0) {
          fly[(i) * fly_offset_3_3 + ( j) * fly_offset_3_2 + ( k) * fly_offset_3_1] = 0;
        }


        if (tid >= (domain_size) * (domain_size)) return;
        i = tid % domain_size;
        j = tid / domain_size;
    //for (int64_t i = 0; i < domain_size; ++i) {
    //  for (int64_t j = 0; j < domain_size; ++j) {
        out[(i) * out_offset_3_3 + ( j) * out_offset_3_2 + ( k) * out_offset_3_1] =
            in[(i) * in_offset_3_3 + ( j) * in_offset_3_2 + ( k) * in_offset_3_1] + (flx[(i - 1) * flx_offset_3_3 + ( j) * flx_offset_3_2 + ( k) * flx_offset_3_1] - flx[(i) * flx_offset_3_3 + ( j) * flx_offset_3_2 + ( k) * flx_offset_3_1] + fly[(i) * fly_offset_3_3 + ( j - 1) * fly_offset_3_2 + ( k) * fly_offset_3_1] - fly[(i) * fly_offset_3_3 + ( j) * fly_offset_3_2 + ( k) * fly_offset_3_1]) * mask[(i) * mask_offset_3_3 + ( j) * mask_offset_3_2 + ( k) * mask_offset_3_1];
}


void hdiffsa_fullfusion(double *out,int64_t out_offset_3_3, int64_t out_offset_3_2, int64_t out_offset_3_1,  double *in,int64_t in_offset_3_3, int64_t in_offset_3_2, int64_t in_offset_3_1,  double *mask,int64_t mask_offset_3_3, int64_t mask_offset_3_2, int64_t mask_offset_3_1,  double* crlato,
              double* crlatu, double *lap,int64_t lap_offset_3_3, int64_t lap_offset_3_2, int64_t lap_offset_3_1, double *flx,int64_t flx_offset_3_3, int64_t flx_offset_3_2, int64_t flx_offset_3_1, double *fly,int64_t fly_offset_3_3, int64_t fly_offset_3_2, int64_t fly_offset_3_1) {

    auto size = (domain_size + 1) * (domain_size + 1) * (domain_height);
    int blocksize = 64;
    hdiffsa_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (
         out, out_offset_3_3,  out_offset_3_2,  out_offset_3_1,  in, in_offset_3_3,  in_offset_3_2,  in_offset_3_1,  mask, mask_offset_3_3,  mask_offset_3_2,  mask_offset_3_1,   crlato,
               crlatu, lap, lap_offset_3_3,  lap_offset_3_2,  lap_offset_3_1, flx, flx_offset_3_3,  flx_offset_3_2,  flx_offset_3_1, fly, fly_offset_3_3,  fly_offset_3_2,  fly_offset_3_1,
            domain_size, domain_height
            );
}

void hdiffsa_fullfusion2(double *out,int64_t out_offset_3_3, int64_t out_offset_3_2, int64_t out_offset_3_1,  double *in,int64_t in_offset_3_3, int64_t in_offset_3_2, int64_t in_offset_3_1,  double *mask,int64_t mask_offset_3_3, int64_t mask_offset_3_2, int64_t mask_offset_3_1,  double* crlato,
              double* crlatu, double *lap,int64_t lap_offset_3_3, int64_t lap_offset_3_2, int64_t lap_offset_3_1, double *flx,int64_t flx_offset_3_3, int64_t flx_offset_3_2, int64_t flx_offset_3_1, double *fly,int64_t fly_offset_3_3, int64_t fly_offset_3_2, int64_t fly_offset_3_1) {

    auto size = (domain_size + 1) * (domain_size + 1) * (domain_height);
    int blocksize = 512;
    hdiffsa_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (
         out, out_offset_3_3,  out_offset_3_2,  out_offset_3_1,  in, in_offset_3_3,  in_offset_3_2,  in_offset_3_1,  mask, mask_offset_3_3,  mask_offset_3_2,  mask_offset_3_1,   crlato,
               crlatu, lap, lap_offset_3_3,  lap_offset_3_2,  lap_offset_3_1, flx, flx_offset_3_3,  flx_offset_3_2,  flx_offset_3_1, fly, fly_offset_3_3,  fly_offset_3_2,  fly_offset_3_1,
            domain_size, domain_height
            );
}


#endif  // HDIFFSA_H
