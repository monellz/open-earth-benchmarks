#ifndef FASTWAVESUV_H
#define FASTWAVESUV_H

#ifndef MANUAL_FUSION
void fastwavesuv(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage3D& utens,
                 const Storage3D& vtens, const Storage3D& wgtfac, const Storage3D& ppuv, const Storage3D& hhl,
                 const Storage3D& rho, const Storage1D& fx, Storage3D& ppgk, Storage3D& ppgc, Storage3D& ppgu,
                 Storage3D& ppgv, const ElementType edadlat, const ElementType dt) {
  for (int64_t i = 0; i < domain_size + 1; ++i) {
    for (int64_t j = 0; j < domain_size + 1; ++j) {
      for (int64_t k = 0; k < domain_height + 1; ++k) {
        ppgk(i, j, k) = wgtfac(i, j, k) * ppuv(i, j, k) + (ElementType(1.0) - wgtfac(i, j, k)) * ppuv(i, j, k - 1);
      }
    }
  }

  for (int64_t i = 0; i < domain_size + 1; ++i) {
    for (int64_t j = 0; j < domain_size + 1; ++j) {
      for (int64_t k = 0; k < domain_height; ++k) {
        ppgc(i, j, k) = ppgk(i, j, k + 1) - ppgk(i, j, k);
      }
    }
  }

  for (int64_t i = 0; i < domain_size; ++i) {
    for (int64_t j = 0; j < domain_size; ++j) {
      for (int64_t k = 0; k < domain_height; ++k) {
        ppgu(i, j, k) = (ppuv(i + 1, j, k) - ppuv(i, j, k)) +
                        (ppgc(i + 1, j, k) + ppgc(i, j, k)) * ElementType(0.5) *
                            ((hhl(i, j, k + 1) + hhl(i, j, k)) - (hhl(i + 1, j, k + 1) + hhl(i + 1, j, k))) /
                            ((hhl(i, j, k + 1) - hhl(i, j, k)) + (hhl(i + 1, j, k + 1) - hhl(i + 1, j, k)));
      }
    }
  }

  for (int64_t i = 0; i < domain_size; ++i) {
    for (int64_t j = 0; j < domain_size; ++j) {
      for (int64_t k = 0; k < domain_height; ++k) {
        ppgv(i, j, k) = (ppuv(i, j + 1, k) - ppuv(i, j, k)) +
                        (ppgc(i, j + 1, k) + ppgc(i, j, k)) * ElementType(0.5) *
                            ((hhl(i, j, k + 1) + hhl(i, j, k)) - (hhl(i, j + 1, k + 1) + hhl(i, j + 1, k))) /
                            ((hhl(i, j, k + 1) - hhl(i, j, k)) + (hhl(i, j + 1, k + 1) - hhl(i, j + 1, k)));
      }
    }
  }

  for (int64_t i = 0; i < domain_size; ++i) {
    for (int64_t j = 0; j < domain_size; ++j) {
      for (int64_t k = 0; k < domain_height; ++k) {
        uout(i, j, k) =
            uin(i, j, k) +
            dt * (utens(i, j, k) - ppgu(i, j, k) * (ElementType(2.0) * fx(j) / (rho(i + 1, j, k) + rho(i, j, k))));
      }
    }
  }

  for (int64_t i = 0; i < domain_size; ++i) {
    for (int64_t j = 0; j < domain_size; ++j) {
      for (int64_t k = 0; k < domain_height; ++k) {
        vout(i, j, k) =
            vin(i, j, k) +
            dt * (vtens(i, j, k) - ppgv(i, j, k) * (ElementType(2.0) * edadlat / (rho(i, j + 1, k) + rho(i, j, k))));
      }
    }
  }
}

#else
void fastwavesuv(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage3D& utens,
                 const Storage3D& vtens, const Storage3D& wgtfac, const Storage3D& ppuv, const Storage3D& hhl,
                 const Storage3D& rho, const Storage1D& fx, Storage3D& ppgk, Storage3D& ppgc, Storage3D& ppgu,
                 Storage3D& ppgv, const ElementType edadlat, const ElementType dt) {
  for (int64_t i = 0; i < domain_size; ++i) {
    for (int64_t j = 0; j < domain_size; ++j) {
      for (int64_t k = 0; k < domain_height; ++k) {
        auto _ppgk_ijk = wgtfac(i, j, k) * ppuv(i, j, k) + (ElementType(1.0) - wgtfac(i, j, k)) * ppuv(i, j, k - 1);
        auto _ppgk_i1jk = wgtfac(i + 1, j, k) * ppuv(i + 1, j, k) + (ElementType(1.0) - wgtfac(i + 1, j, k)) * ppuv(i + 1, j, k - 1);
        auto _ppgk_ij1k = wgtfac(i, j + 1, k) * ppuv(i, j + 1, k) + (ElementType(1.0) - wgtfac(i, j + 1, k)) * ppuv(i, j + 1, k - 1);
        auto _ppgk_ijk1 = wgtfac(i, j, k + 1) * ppuv(i, j, k + 1) + (ElementType(1.0) - wgtfac(i, j, k + 1)) * ppuv(i, j, k);
        auto _ppgk_i1jk1 = wgtfac(i + 1, j, k + 1) * ppuv(i + 1, j, k + 1) + (ElementType(1.0) - wgtfac(i + 1, j, k + 1)) * ppuv(i + 1, j, k);
        auto _ppgk_ij1k1 = wgtfac(i, j + 1, k + 1) * ppuv(i, j + 1, k + 1) + (ElementType(1.0) - wgtfac(i, j + 1, k + 1)) * ppuv(i, j + 1, k);

        auto _ppgc_ijk = _ppgk_ijk1 - _ppgk_ijk;
        auto _ppgc_i1jk = _ppgk_i1jk1 - _ppgk_i1jk;
        auto _ppgc_ij1k = _ppgk_ij1k1 - _ppgk_ij1k;

        auto _ppgu = (ppuv(i + 1, j, k) - ppuv(i, j, k)) +
                     (_ppgc_i1jk + _ppgc_ijk) * ElementType(0.5) *
                         ((hhl(i, j, k + 1) + hhl(i, j, k)) - (hhl(i + 1, j, k + 1) + hhl(i + 1, j, k))) /
                         ((hhl(i, j, k + 1) - hhl(i, j, k)) + (hhl(i + 1, j, k + 1) - hhl(i + 1, j, k)));

        auto _ppgv = (ppuv(i, j + 1, k) - ppuv(i, j, k)) +
                     (_ppgc_ij1k + _ppgc_ijk) * ElementType(0.5) *
                         ((hhl(i, j, k + 1) + hhl(i, j, k)) - (hhl(i, j + 1, k + 1) + hhl(i, j + 1, k))) /
                         ((hhl(i, j, k + 1) - hhl(i, j, k)) + (hhl(i, j + 1, k + 1) - hhl(i, j + 1, k)));

        uout(i, j, k) = uin(i, j, k) +
                        dt * (utens(i, j, k) - _ppgu * (ElementType(2.0) * fx(j) / (rho(i + 1, j, k) + rho(i, j, k))));

        vout(i, j, k) = vin(i, j, k) + dt * (vtens(i, j, k) -
                                             _ppgv * (ElementType(2.0) * edadlat / (rho(i, j + 1, k) + rho(i, j, k))));
      }
    }
  }
}
#endif

#endif  // FASTWAVESUV_H
