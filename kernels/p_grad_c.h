#ifndef P_GRAD_C_H
#define P_GRAD_C_H

void p_grad_c(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage3D& rdxc,
              const Storage3D& rdyc, const Storage3D& delpc, const Storage3D& gz, const Storage3D& pkc, Storage3D& wk,
              const ElementType dt2) {
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        wk(i, j, k) = delpc(i, j, k);
      }
    }

    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        uout(i, j, k) =
            (uin(i, j, k) + (((dt2 * rdxc(i, j, k)) / (wk(i - 1, j, k) + wk(i, j, k))) *
                             (((gz(i - 1, j, k + 1) - gz(i, j, k)) * (pkc(i, j, k + 1) - pkc(i - 1, j, k))) +
                              ((gz(i - 1, j, k) - gz(i, j, k + 1)) * (pkc(i - 1, j, k + 1) - pkc(i, j, k))))));
        vout(i, j, k) =
            (vin(i, j, k) + (((dt2 * rdyc(i, j, k)) / (wk(i, j - 1, k) + wk(i, j, k))) *
                             (((gz(i, j - 1, k + 1) - gz(i, j, k)) * (pkc(i, j, k + 1) - pkc(i, j - 1, k))) +
                              ((gz(i, j - 1, k) - gz(i, j, k + 1)) * (pkc(i, j - 1, k + 1) - pkc(i, j, k))))));
      }
    }
  }
}

void p_grad_c_fullfusion(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage3D& rdxc,
              const Storage3D& rdyc, const Storage3D& delpc, const Storage3D& gz, const Storage3D& pkc, Storage3D& wk,
              const ElementType dt2) {
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _wk_ijk = delpc(i, j, k);
        auto _wk_im1jk = delpc(i - 1, j, k);
        auto _wk_ijm1k = delpc(i, j - 1, k);
        uout(i, j, k) =
            (uin(i, j, k) + (((dt2 * rdxc(i, j, k)) / (_wk_im1jk + _wk_ijk)) *
                             (((gz(i - 1, j, k + 1) - gz(i, j, k)) * (pkc(i, j, k + 1) - pkc(i - 1, j, k))) +
                              ((gz(i - 1, j, k) - gz(i, j, k + 1)) * (pkc(i - 1, j, k + 1) - pkc(i, j, k))))));
        vout(i, j, k) =
            (vin(i, j, k) + (((dt2 * rdyc(i, j, k)) / (_wk_ijm1k + _wk_ijk)) *
                             (((gz(i, j - 1, k + 1) - gz(i, j, k)) * (pkc(i, j, k + 1) - pkc(i, j - 1, k))) +
                              ((gz(i, j - 1, k) - gz(i, j, k + 1)) * (pkc(i, j - 1, k + 1) - pkc(i, j, k))))));
      }
    }
  }
}

void p_grad_c_partialfusion(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage3D& rdxc,
              const Storage3D& rdyc, const Storage3D& delpc, const Storage3D& gz, const Storage3D& pkc, Storage3D& wk,
              const ElementType dt2) {
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _wk_ijk = delpc(i, j, k);
        auto _wk_im1jk = delpc(i - 1, j, k);
        uout(i, j, k) =
            (uin(i, j, k) + (((dt2 * rdxc(i, j, k)) / (_wk_im1jk + _wk_ijk)) *
                             (((gz(i - 1, j, k + 1) - gz(i, j, k)) * (pkc(i, j, k + 1) - pkc(i - 1, j, k))) +
                              ((gz(i - 1, j, k) - gz(i, j, k + 1)) * (pkc(i - 1, j, k + 1) - pkc(i, j, k))))));
      }
    }
  }
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _wk_ijk = delpc(i, j, k);
        auto _wk_ijm1k = delpc(i, j - 1, k);
        vout(i, j, k) =
            (vin(i, j, k) + (((dt2 * rdyc(i, j, k)) / (_wk_ijm1k + _wk_ijk)) *
                             (((gz(i, j - 1, k + 1) - gz(i, j, k)) * (pkc(i, j, k + 1) - pkc(i, j - 1, k))) +
                              ((gz(i, j - 1, k) - gz(i, j, k + 1)) * (pkc(i, j - 1, k + 1) - pkc(i, j, k))))));
      }
    }
  }
}

void p_grad_c_openmp(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage3D& rdxc,
              const Storage3D& rdyc, const Storage3D& delpc, const Storage3D& gz, const Storage3D& pkc, Storage3D& wk,
              const ElementType dt2) {
  #pragma omp parallel for
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _wk_ijk = delpc(i, j, k);
        auto _wk_im1jk = delpc(i - 1, j, k);
        auto _wk_ijm1k = delpc(i, j - 1, k);
        uout(i, j, k) =
            (uin(i, j, k) + (((dt2 * rdxc(i, j, k)) / (_wk_im1jk + _wk_ijk)) *
                             (((gz(i - 1, j, k + 1) - gz(i, j, k)) * (pkc(i, j, k + 1) - pkc(i - 1, j, k))) +
                              ((gz(i - 1, j, k) - gz(i, j, k + 1)) * (pkc(i - 1, j, k + 1) - pkc(i, j, k))))));
        vout(i, j, k) =
            (vin(i, j, k) + (((dt2 * rdyc(i, j, k)) / (_wk_ijm1k + _wk_ijk)) *
                             (((gz(i, j - 1, k + 1) - gz(i, j, k)) * (pkc(i, j, k + 1) - pkc(i, j - 1, k))) +
                              ((gz(i, j - 1, k) - gz(i, j, k + 1)) * (pkc(i, j - 1, k + 1) - pkc(i, j, k))))));
      }
    }
  }
}

#endif  // P_GRAD_C_H
