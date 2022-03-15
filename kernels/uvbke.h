#ifndef UVBKE_H
#define UVBKE_H

__global__ void uvbke_fullfusion_k1(double *ub,int64_t ub_offset_3_3, int64_t ub_offset_3_2, int64_t ub_offset_3_1, double *vb,int64_t vb_offset_3_3, int64_t vb_offset_3_2, int64_t vb_offset_3_1,  double *uc,int64_t uc_offset_3_3, int64_t uc_offset_3_2, int64_t uc_offset_3_1,  double *vc,int64_t vc_offset_3_3, int64_t vc_offset_3_2, int64_t vc_offset_3_1,  double *cosa,int64_t cosa_offset_3_3, int64_t cosa_offset_3_2, int64_t cosa_offset_3_1,
            double *rsina,int64_t rsina_offset_3_3, int64_t rsina_offset_3_2, int64_t rsina_offset_3_1, int64_t domain_size, int64_t domain_height) {
  // for (int64_t k = 0; k < domain_height; ++k) {
  //   for (int64_t i = 0; i < domain_size; ++i) {
  //     for (int64_t j = 0; j < domain_size; ++j) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= (domain_size) * (domain_size) * (domain_height)) return;
        int k = tid / (domain_size) / (domain_size);
        int i = (tid % ((domain_size) * (domain_size))) / (domain_size);
        int j = tid % (domain_size);
        ub[(i) * ub_offset_3_3 + ( j) * ub_offset_3_2 + ( k) * ub_offset_3_1] = ((dt5 * ((uc[(i) * uc_offset_3_3 + ( j - 1) * uc_offset_3_2 + ( k) * uc_offset_3_1] + uc[(i) * uc_offset_3_3 + ( j) * uc_offset_3_2 + ( k) * uc_offset_3_1]) - ((vc[(i - 1) * vc_offset_3_3 + ( j) * vc_offset_3_2 + ( k) * vc_offset_3_1] + vc[(i) * vc_offset_3_3 + ( j) * vc_offset_3_2 + ( k) * vc_offset_3_1]) * cosa[(i) * cosa_offset_3_3 + ( j) * cosa_offset_3_2 + ( k) * cosa_offset_3_1]))) *
                       rsina[(i) * rsina_offset_3_3 + ( j) * rsina_offset_3_2 + ( k) * rsina_offset_3_1]);

        vb[(i) * vb_offset_3_3 + ( j) * vb_offset_3_2 + ( k) * vb_offset_3_1] = ((dt5 * ((vc[(i - 1) * vc_offset_3_3 + ( j) * vc_offset_3_2 + ( k) * vc_offset_3_1] + vc[(i) * vc_offset_3_3 + ( j) * vc_offset_3_2 + ( k) * vc_offset_3_1]) - ((uc[(i) * uc_offset_3_3 + ( j - 1) * uc_offset_3_2 + ( k) * uc_offset_3_1] + uc[(i) * uc_offset_3_3 + ( j) * uc_offset_3_2 + ( k) * uc_offset_3_1]) * cosa[(i) * cosa_offset_3_3 + ( j) * cosa_offset_3_2 + ( k) * cosa_offset_3_1]))) *
                       rsina[(i) * rsina_offset_3_3 + ( j) * rsina_offset_3_2 + ( k) * rsina_offset_3_1]);
  //    }
  //  }
  //}
}


void uvbke_fullfusion(double *ub,int64_t ub_offset_3_3, int64_t ub_offset_3_2, int64_t ub_offset_3_1, double *vb,int64_t vb_offset_3_3, int64_t vb_offset_3_2, int64_t vb_offset_3_1,  double *uc,int64_t uc_offset_3_3, int64_t uc_offset_3_2, int64_t uc_offset_3_1,  double *vc,int64_t vc_offset_3_3, int64_t vc_offset_3_2, int64_t vc_offset_3_1,  double *cosa,int64_t cosa_offset_3_3, int64_t cosa_offset_3_2, int64_t cosa_offset_3_1,
            double *rsina,int64_t rsina_offset_3_3, int64_t rsina_offset_3_2, int64_t rsina_offset_3_1) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 64;
    uvbke_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (ub, ub_offset_3_3,  ub_offset_3_2,  ub_offset_3_1, vb, vb_offset_3_3,  vb_offset_3_2,  vb_offset_3_1,  uc, uc_offset_3_3,  uc_offset_3_2,  uc_offset_3_1,  vc, vc_offset_3_3,  vc_offset_3_2,  vc_offset_3_1,  cosa, cosa_offset_3_3,  cosa_offset_3_2,  cosa_offset_3_1,
            rsina, rsina_offset_3_3,  rsina_offset_3_2,  rsina_offset_3_1,  domain_size,  domain_height);
}

void uvbke_fullfusion2(double *ub,int64_t ub_offset_3_3, int64_t ub_offset_3_2, int64_t ub_offset_3_1, double *vb,int64_t vb_offset_3_3, int64_t vb_offset_3_2, int64_t vb_offset_3_1,  double *uc,int64_t uc_offset_3_3, int64_t uc_offset_3_2, int64_t uc_offset_3_1,  double *vc,int64_t vc_offset_3_3, int64_t vc_offset_3_2, int64_t vc_offset_3_1,  double *cosa,int64_t cosa_offset_3_3, int64_t cosa_offset_3_2, int64_t cosa_offset_3_1,
            double *rsina,int64_t rsina_offset_3_3, int64_t rsina_offset_3_2, int64_t rsina_offset_3_1) {
    auto size = (domain_size) * (domain_size) * (domain_height);
    int blocksize = 512;
    uvbke_fullfusion_k1<<< (size + blocksize - 1) / blocksize, blocksize >>>
        (ub, ub_offset_3_3,  ub_offset_3_2,  ub_offset_3_1, vb, vb_offset_3_3,  vb_offset_3_2,  vb_offset_3_1,  uc, uc_offset_3_3,  uc_offset_3_2,  uc_offset_3_1,  vc, vc_offset_3_3,  vc_offset_3_2,  vc_offset_3_1,  cosa, cosa_offset_3_3,  cosa_offset_3_2,  cosa_offset_3_1,
            rsina, rsina_offset_3_3,  rsina_offset_3_2,  rsina_offset_3_1,  domain_size,  domain_height);
}

#endif  // UVBKE_H
