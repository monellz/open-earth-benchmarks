#ifndef FVTP2D_H
#define FVTP2D_H

void fvtp2d(Storage3D& q_i, Storage3D& q_j, Storage3D& fx1, Storage3D& fx2, Storage3D& fy1, Storage3D& fy2,
            const Storage3D& q, const Storage3D& crx, const Storage3D& cry, const Storage3D& ra_x,
            const Storage3D& ra_y, const Storage3D& xfx, const Storage3D& yfx, const Storage3D& area, Storage3D& fxx,
            Storage3D& fyy, Storage3D& al, Storage3D& almq, Storage3D& br, Storage3D& b0, Storage3D& smt5) {
  // fy2
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = -3; i < domain_size + 2; ++i) {
      for (int64_t j = -1; j < domain_size + 2; ++j) {
        al(i, j, k) = ((p1 * (q(i, j - 1, k) + q(i, j, k))) + (p2 * (q(i, j - 2, k) + q(i, j + 1, k))));
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = -3; i < domain_size + 2; ++i) {
      for (int64_t j = -1; j < domain_size + 1; ++j) {
        almq(i, j, k) = (al(i, j, k) - q(i, j, k));
        br(i, j, k) = (al(i, j + 1, k) - q(i, j, k));
        b0(i, j, k) = (almq(i, j, k) + br(i, j, k));
        smt5(i, j, k) = ((almq(i, j, k) * br(i, j, k)) < int64_t(0));
      }
    }
    for (int64_t i = -3; i < domain_size + 2; ++i) {
      for (int64_t j = 0; j < domain_size + 1; ++j) {
        ElementType smt5tmp = (smt5(i, j - 1, k) + (smt5(i, j, k) * (smt5(i, j - 1, k) == int64_t(0))));
        ElementType crytmp =
            ((cry(i, j, k) > ElementType(0.))
                 ? ((ElementType(1.) - cry(i, j, k)) * (br(i, j - 1, k) - (cry(i, j, k) * b0(i, j - 1, k))))
                 : ((ElementType(1.) + cry(i, j, k)) * (almq(i, j, k) + (cry(i, j, k) * b0(i, j, k)))));
        fy2(i, j, k) = ((cry(i, j, k) > ElementType(0.)) ? (q(i, j - 1, k) + (crytmp * smt5tmp))
                                                         : (q(i, j, k) + (crytmp * smt5tmp)));
      }
    }
  }

  // q_i
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = -3; i < domain_size + 2; ++i) {
      for (int64_t j = 0; j < domain_size + 1; ++j) {
        fyy(i, j, k) = (yfx(i, j, k) * fy2(i, j, k));
      }
    }
    for (int64_t i = -3; i < domain_size + 2; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        q_i(i, j, k) = ((((q(i, j, k) * area(i, j, k)) + fyy(i, j, k)) - fyy(i, j + 1, k)) / ra_y(i, j, k));
      }
    }
  }

  // fx1
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = -1; i < domain_size + 1; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        al(i, j, k) = ((p1 * (q_i(i - 1, j, k) + q_i(i, j, k))) + (p2 * (q_i(i - 2, j, k) + q_i(i + 1, j, k))));
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = -1; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        almq(i, j, k) = (al(i, j, k) - q_i(i, j, k));
        br(i, j, k) = (al(i + 1, j, k) - q_i(i, j, k));
        b0(i, j, k) = (almq(i, j, k) + br(i, j, k));
        smt5(i, j, k) = ((almq(i, j, k) * br(i, j, k)) < int64_t(0));
      }
    }
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        ElementType smt5tmp = (smt5(i - 1, j, k) + (smt5(i, j, k) * (smt5(i - 1, j, k) == int64_t(0))));
        ElementType crxtmp =
            ((crx(i, j, k) > ElementType(0.))
                 ? ((ElementType(1.) - crx(i, j, k)) * (br(i - 1, j, k) - (crx(i, j, k) * b0(i - 1, j, k))))
                 : ((ElementType(1.) + crx(i, j, k)) * (almq(i, j, k) + (crx(i, j, k) * b0(i, j, k)))));
        fx1(i, j, k) = ((crx(i, j, k) > ElementType(0.)) ? (q_i(i - 1, j, k) + (crxtmp * smt5tmp))
                                                         : (q_i(i, j, k) + (crxtmp * smt5tmp)));
      }
    }
  }

  // fx2
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = -1; i < domain_size + 2; ++i) {
      for (int64_t j = -3; j < domain_size + 2; ++j) {
        al(i, j, k) = ((p1 * (q(i - 1, j, k) + q(i, j, k))) + (p2 * (q(i - 2, j, k) + q(i + 1, j, k))));
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = -1; i < domain_size + 1; ++i) {
      for (int64_t j = -3; j < domain_size + 2; ++j) {
        almq(i, j, k) = (al(i, j, k) - q(i, j, k));
        br(i, j, k) = (al(i + 1, j, k) - q(i, j, k));
        b0(i, j, k) = (almq(i, j, k) + br(i, j, k));
        smt5(i, j, k) = ((almq(i, j, k) * br(i, j, k)) < int64_t(0));
      }
    }
    for (int64_t i = 0; i < domain_size + 1; ++i) {
      for (int64_t j = -3; j < domain_size + 2; ++j) {
        ElementType smt5tmp = (smt5(i - 1, j, k) + (smt5(i, j, k) * (smt5(i - 1, j, k) == int64_t(0))));
        ElementType crxtmp =
            ((crx(i, j, k) > ElementType(0.))
                 ? ((ElementType(1.) - crx(i, j, k)) * (br(i - 1, j, k) - (crx(i, j, k) * b0(i - 1, j, k))))
                 : ((ElementType(1.) + crx(i, j, k)) * (almq(i, j, k) + (crx(i, j, k) * b0(i, j, k)))));
        fx2(i, j, k) = ((crx(i, j, k) > ElementType(0.)) ? (q(i - 1, j, k) + (crxtmp * smt5tmp))
                                                         : (q(i, j, k) + (crxtmp * smt5tmp)));
      }
    }
  }

  // q_j
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size + 1; ++i) {
      for (int64_t j = -3; j < domain_size + 2; ++j) {
        fxx(i, j, k) = (xfx(i, j, k) * fx2(i, j, k));
      }
    }
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = -3; j < domain_size + 2; ++j) {
        q_j(i, j, k) = ((((q(i, j, k) * area(i, j, k)) + fxx(i, j, k)) - fxx(i + 1, j, k)) / ra_x(i, j, k));
      }
    }
  }

  // fy1
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = -1; j < domain_size + 1; ++j) {
        al(i, j, k) = ((p1 * (q_j(i, j - 1, k) + q_j(i, j, k))) + (p2 * (q_j(i, j - 2, k) + q_j(i, j + 1, k))));
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = -1; j < domain_size; ++j) {
        almq(i, j, k) = (al(i, j, k) - q_j(i, j, k));
        br(i, j, k) = (al(i, j + 1, k) - q_j(i, j, k));
        b0(i, j, k) = (almq(i, j, k) + br(i, j, k));
        smt5(i, j, k) = ((almq(i, j, k) * br(i, j, k)) < int64_t(0));
      }
    }
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        ElementType smt5tmp = (smt5(i, j - 1, k) + (smt5(i, j, k) * (smt5(i, j - 1, k) == int64_t(0))));
        ElementType crytmp =
            ((cry(i, j, k) > ElementType(0.))
                 ? ((ElementType(1.) - cry(i, j, k)) * (br(i, j - 1, k) - (cry(i, j, k) * b0(i, j - 1, k))))
                 : ((ElementType(1.) + cry(i, j, k)) * (almq(i, j, k) + (cry(i, j, k) * b0(i, j, k)))));
        fy1(i, j, k) = ((cry(i, j, k) > ElementType(0.)) ? (q_j(i, j - 1, k) + (crytmp * smt5tmp))
                                                         : (q_j(i, j, k) + (crytmp * smt5tmp)));
      }
    }
  }

  // transport_flux
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        fx1(i, j, k) = ((ElementType(0.5) * (fx1(i, j, k) + fx2(i, j, k))) * xfx(i, j, k));
      }
    }
  }

  // transport_flux
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        fy1(i, j, k) = ((ElementType(0.5) * (fy1(i, j, k) + fy2(i, j, k))) * yfx(i, j, k));
      }
    }
  }
}

void fvtp2d_fullfusion(Storage3D& q_i, Storage3D& q_j, Storage3D& fx1, Storage3D& fx2, Storage3D& fy1, Storage3D& fy2,
            const Storage3D& q, const Storage3D& crx, const Storage3D& cry, const Storage3D& ra_x,
            const Storage3D& ra_y, const Storage3D& xfx, const Storage3D& yfx, const Storage3D& area, Storage3D& fxx,
            Storage3D& fyy, Storage3D& al, Storage3D& almq, Storage3D& br, Storage3D& b0, Storage3D& smt5) {
  // fy2
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = -3; i < domain_size + 2; ++i) {
      for (int64_t j = 0; j < domain_size + 1; ++j) {
        auto _al_ijk = ((p1 * (q(i, j - 1, k) + q(i, j, k))) + (p2 * (q(i, j - 2, k) + q(i, j + 1, k))));
        auto _al_ij1k = ((p1 * (q(i, j, k) + q(i, j + 1, k))) + (p2 * (q(i, j - 1, k) + q(i, j + 2, k))));
        auto _al_ijm1k = ((p1 * (q(i, j - 2, k) + q(i, j - 1, k))) + (p2 * (q(i, j - 3, k) + q(i, j, k))));

        auto _almq_ijk = (_al_ijk - q(i, j, k));
        auto _almq_ijm1k = (_al_ijm1k - q(i, j - 1, k));
        auto _br_ijk = (_al_ij1k - q(i, j, k));
        auto _br_ijm1k = (_al_ijk - q(i, j - 1, k));
        auto _b0_ijk = (_almq_ijk + br(i, j, k));
        auto _b0_ijm1k = (_almq_ijm1k + br(i, j - 1, k));
        auto _smt5_ijk = ((_almq_ijk * _br_ijk) < int64_t(0));
        auto _smt5_ijm1k = ((_almq_ijm1k * _br_ijm1k) < int64_t(0));


        ElementType smt5tmp = (_smt5_ijm1k + (_smt5_ijk * (_smt5_ijm1k == int64_t(0))));
        ElementType crytmp =
            ((cry(i, j, k) > ElementType(0.))
                 ? ((ElementType(1.) - cry(i, j, k)) * (_br_ijm1k - (cry(i, j, k) * _b0_ijm1k)))
                 : ((ElementType(1.) + cry(i, j, k)) * (_almq_ijk + (cry(i, j, k) * _b0_ijk))));
        fy2(i, j, k) = ((cry(i, j, k) > ElementType(0.)) ? (q(i, j - 1, k) + (crytmp * smt5tmp))
                                                         : (q(i, j, k) + (crytmp * smt5tmp)));
      }
    }
  }

  // q_i
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = -3; i < domain_size + 2; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _fyy_ijk = (yfx(i, j, k) * fy2(i, j, k));
        auto _fyy_ij1k = (yfx(i, j + 1, k) * fy2(i, j + 1, k));
        q_i(i, j, k) = ((((q(i, j, k) * area(i, j, k)) + _fyy_ijk) - _fyy_ij1k) / ra_y(i, j, k));
      }
    }
  }

  // fx1
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _al_ijk = ((p1 * (q_i(i - 1, j, k) + q_i(i, j, k))) + (p2 * (q_i(i - 2, j, k) + q_i(i + 1, j, k))));
        auto _al_i1jk = ((p1 * (q_i(i, j, k) + q_i(i + 1, j, k))) + (p2 * (q_i(i - 1, j, k) + q_i(i + 2, j, k))));
        auto _al_im1jk = ((p1 * (q_i(i - 2, j, k) + q_i(i - 1, j, k))) + (p2 * (q_i(i - 3, j, k) + q_i(i, j, k))));
        auto _almq_ijk = (_al_ijk - q_i(i, j, k));
        auto _almq_im1jk = (_al_im1jk - q_i(i - 1, j, k));
        auto _br_ijk = (_al_i1jk - q_i(i, j, k));
        auto _br_im1jk = (_al_ijk - q_i(i - 1, j, k));
        auto _b0_ijk = (_almq_ijk + _br_ijk);
        auto _b0_im1jk = (_almq_im1jk + _br_im1jk);
        auto _smt5_ijk = ((_almq_ijk * _br_ijk) < int64_t(0));
        auto _smt5_im1jk = ((_almq_im1jk * _br_im1jk) < int64_t(0));



        ElementType smt5tmp = (_smt5_im1jk + (_smt5_ijk * (_smt5_im1jk == int64_t(0))));
        ElementType crxtmp =
            ((crx(i, j, k) > ElementType(0.))
                 ? ((ElementType(1.) - crx(i, j, k)) * (_br_im1jk - (crx(i, j, k) * _b0_im1jk)))
                 : ((ElementType(1.) + crx(i, j, k)) * (_almq_ijk + (crx(i, j, k) * _b0_ijk))));
        fx1(i, j, k) = ((crx(i, j, k) > ElementType(0.)) ? (q_i(i - 1, j, k) + (crxtmp * smt5tmp))
                                                         : (q_i(i, j, k) + (crxtmp * smt5tmp)));
      }
    }
  }

  // fx2
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size + 1; ++i) {
      for (int64_t j = -3; j < domain_size + 2; ++j) {
        auto _al_ijk = ((p1 * (q(i - 1, j, k) + q(i, j, k))) + (p2 * (q(i - 2, j, k) + q(i + 1, j, k))));
        auto _al_i1jk = ((p1 * (q(i, j, k) + q(i + 1, j, k))) + (p2 * (q(i - 1, j, k) + q(i + 2, j, k))));
        auto _al_im1jk = ((p1 * (q(i - 2, j, k) + q(i - 1, j, k))) + (p2 * (q(i - 3, j, k) + q(i, j, k))));
        auto _almq_ijk = (_al_ijk - q(i, j, k));
        auto _almq_im1jk = (_al_im1jk - q(i - 1, j, k));
        auto _br_ijk = (_al_i1jk - q(i, j, k));
        auto _br_im1jk = (_al_ijk - q(i - 1, j, k));
        auto _b0_ijk = (_almq_ijk + _br_ijk);
        auto _b0_im1jk = (_almq_im1jk + _br_im1jk);
        auto _smt5_ijk = ((_almq_ijk * _br_ijk) < int64_t(0));
        auto _smt5_im1jk = ((_almq_im1jk * _br_im1jk) < int64_t(0));

        ElementType smt5tmp = (_smt5_im1jk + (_smt5_ijk * (_smt5_im1jk == int64_t(0))));
        ElementType crxtmp =
            ((crx(i, j, k) > ElementType(0.))
                 ? ((ElementType(1.) - crx(i, j, k)) * (_br_im1jk - (crx(i, j, k) * _b0_im1jk)))
                 : ((ElementType(1.) + crx(i, j, k)) * (_almq_ijk + (crx(i, j, k) * _b0_ijk))));
        fx2(i, j, k) = ((crx(i, j, k) > ElementType(0.)) ? (q(i - 1, j, k) + (crxtmp * smt5tmp))
                                                         : (q(i, j, k) + (crxtmp * smt5tmp)));
      }
    }
  }

  // q_j
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = -3; j < domain_size + 2; ++j) {
        auto _fxx_ijk = (xfx(i, j, k) * fx2(i, j, k));
        auto _fxx_i1jk = (xfx(i + 1, j, k) * fx2(i + 1, j, k));
        q_j(i, j, k) = ((((q(i, j, k) * area(i, j, k)) + _fxx_ijk) - _fxx_i1jk) / ra_x(i, j, k));
      }
    }
  }

  // fy1
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _al_ijk = ((p1 * (q_j(i, j - 1, k) + q_j(i, j, k))) + (p2 * (q_j(i, j - 2, k) + q_j(i, j + 1, k))));
        auto _al_ij1k = ((p1 * (q_j(i, j, k) + q_j(i, j + 1, k))) + (p2 * (q_j(i, j - 1, k) + q_j(i, j + 2, k))));
        auto _al_ijm1k = ((p1 * (q_j(i, j - 2, k) + q_j(i, j - 1, k))) + (p2 * (q_j(i, j - 3, k) + q_j(i, j, k))));
        auto _almq_ijk = (_al_ijk - q_j(i, j, k));
        auto _almq_ijm1k = (_al_ijm1k - q_j(i, j - 1, k));
        auto _br_ijk = (_al_ij1k - q_j(i, j, k));
        auto _br_ijm1k = (_al_ijk - q_j(i, j - 1, k));
        auto _b0_ijk = (_almq_ijk + _br_ijk);
        auto _b0_ijm1k = (_almq_ijm1k + _br_ijm1k);
        auto _smt5_ijk = ((_almq_ijk * _br_ijk) < int64_t(0));
        auto _smt5_ijm1k = ((_almq_ijm1k * _br_ijm1k) < int64_t(0));



        ElementType smt5tmp = (_smt5_ijm1k + (_smt5_ijk * (_smt5_ijm1k == int64_t(0))));
        ElementType crytmp =
            ((cry(i, j, k) > ElementType(0.))
                 ? ((ElementType(1.) - cry(i, j, k)) * (_br_ijm1k - (cry(i, j, k) * _b0_ijm1k)))
                 : ((ElementType(1.) + cry(i, j, k)) * (_almq_ijk + (cry(i, j, k) * _b0_ijk))));
        fy1(i, j, k) = ((cry(i, j, k) > ElementType(0.)) ? (q_j(i, j - 1, k) + (crytmp * smt5tmp))
                                                         : (q_j(i, j, k) + (crytmp * smt5tmp)));
      }
    }
  }

  // transport_flux
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        fx1(i, j, k) = ((ElementType(0.5) * (fx1(i, j, k) + fx2(i, j, k))) * xfx(i, j, k));
      }
    }
  }

  // transport_flux
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        fy1(i, j, k) = ((ElementType(0.5) * (fy1(i, j, k) + fy2(i, j, k))) * yfx(i, j, k));
      }
    }
  }
}

#endif  // FVTP2D_H
