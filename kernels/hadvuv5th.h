#ifndef HADVUV5TH_H
#define HADVUV5TH_H

ElementType advectionDriver(const Storage3D field, const int64_t i, const int64_t j, const int64_t k,
                            const ElementType uavg, const ElementType vavg, const ElementType eddlat,
                            const ElementType eddlon) {
  ElementType result_x = 0.0;
  ElementType result_y = 0.0;

  if (uavg > 0) {
    result_x = uavg * (ElementType(1.0) / ElementType(30.0) * field(i - 3, j, k) +
                       ElementType(-1.0) / ElementType(4.0) * field(i - 2, j, k) + field(i - 1, j, k) +
                       ElementType(-1.0) / ElementType(3.0) * field(i, j, k) +
                       ElementType(-1.0) / ElementType(2.0) * field(i + 1, j, k) +
                       ElementType(1.0) / ElementType(20.0) * field(i + 2, j, k));
  } else {
    result_x = -uavg * (ElementType(1.0) / ElementType(20.0) * field(i - 2, j, k) +
                        ElementType(-1.0) / ElementType(2.0) * field(i - 1, j, k) +
                        ElementType(-1.0) / ElementType(3.0) * field(i, j, k) + field(i + 1, j, k) +
                        ElementType(-1.0) / ElementType(4.0) * field(i + 2, j, k) +
                        ElementType(1.0) / ElementType(30.0) * field(i + 3, j, k));
  }

  if (vavg > 0) {
    result_y = vavg * (ElementType(1.0) / ElementType(30.0) * field(i, j - 3, k) +
                       ElementType(-1.0) / ElementType(4.0) * field(i, j - 2, k) + field(i, j - 1, k) +
                       ElementType(-1.0) / ElementType(3.0) * field(i, j, k) +
                       ElementType(-1.0) / ElementType(2.0) * field(i, j + 1, k) +
                       ElementType(1.0) / ElementType(20.0) * field(i, j + 2, k));
  } else {
    result_y = -vavg * (ElementType(1.0) / ElementType(20.0) * field(i, j - 2, k) +
                        ElementType(-1.0) / ElementType(2.0) * field(i, j - 1, k) +
                        ElementType(-1.0) / ElementType(3.0) * field(i, j, k) + field(i, j + 1, k) +
                        ElementType(-1.0) / ElementType(4.0) * field(i, j + 2, k) +
                        ElementType(1.0) / ElementType(30.0) * field(i, j + 3, k));
  }

  return eddlat * result_x + eddlon * result_y;
}

void hadvuv5th(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage1D& acrlat0,
               const Storage1D& acrlat1, const Storage1D& tgrlatda0, const Storage1D& tgrlatda1, Storage3D& uatupos,
               Storage3D& vatupos, Storage3D& uatvpos, Storage3D& vatvpos, Storage3D& uavg, Storage3D& vavg,
               Storage3D& ures, Storage3D& vres, const ElementType eddlat, const ElementType eddlon) {
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        uatupos(i, j, k) = (ElementType(1.0) / ElementType(3.0)) * (uin(i - 1, j, k) + uin(i, j, k) + uin(i + 1, j, k));

        vatupos(i, j, k) =
            ElementType(0.25) * (vin(i + 1, j, k) + vin(i + 1, j - 1, k) + vin(i, j, k) + vin(i, j - 1, k));
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        uavg(i, j, k) = acrlat0(j) * uatupos(i, j, k);
        vavg(i, j, k) = EARTH_RADIUS_RECIP * vatupos(i, j, k);
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        ures(i, j, k) = advectionDriver(uin, i, j, k, uavg(i, j, k), vavg(i, j, k), eddlat, eddlon);
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        uout(i, j, k) = ures(i, j, k) + tgrlatda0(j) * uin(i, j, k) * vatupos(i, j, k);
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        uatvpos(i, j, k) =
            ElementType(0.25) * (uin(i - 1, j, k) + uin(i, j, k) + uin(i, j + 1, k) + uin(i - 1, j + 1, k));

        vatvpos(i, j, k) = ElementType(1.0) / ElementType(3.0) * (vin(i, j - 1, k) + vin(i, j, k) + vin(i, j + 1, k));
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        uavg(i, j, k) = acrlat1(j) * uatvpos(i, j, k);
        vavg(i, j, k) = EARTH_RADIUS_RECIP * vatvpos(i, j, k);
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        vres(i, j, k) = advectionDriver(vin, i, j, k, uavg(i, j, k), vavg(i, j, k), eddlat, eddlon);
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        vout(i, j, k) = vres(i, j, k) - tgrlatda1(j) * uatvpos(i, j, k) * uatvpos(i, j, k);
      }
    }
  }
}

void hadvuv5th_fullfusion(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage1D& acrlat0,
               const Storage1D& acrlat1, const Storage1D& tgrlatda0, const Storage1D& tgrlatda1, Storage3D& uatupos,
               Storage3D& vatupos, Storage3D& uatvpos, Storage3D& vatvpos, Storage3D& uavg, Storage3D& vavg,
               Storage3D& ures, Storage3D& vres, const ElementType eddlat, const ElementType eddlon) {
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _uatupos = (ElementType(1.0) / ElementType(3.0)) * (uin(i - 1, j, k) + uin(i, j, k) + uin(i + 1, j, k));
        auto _vatupos = ElementType(0.25) * (vin(i + 1, j, k) + vin(i + 1, j - 1, k) + vin(i, j, k) + vin(i, j - 1, k));
        auto _uavg = acrlat0(j) * _uatupos;
        auto _vavg = EARTH_RADIUS_RECIP * _vatupos;
        auto _ures = advectionDriver(uin, i, j, k, _uavg, _vavg, eddlat, eddlon);
        uout(i, j, k) = _ures + tgrlatda0(j) * uin(i, j, k) * _vatupos;

        auto _uatvpos = ElementType(0.25) * (uin(i - 1, j, k) + uin(i, j, k) + uin(i, j + 1, k) + uin(i - 1, j + 1, k));
        auto _vatvpos = ElementType(1.0) / ElementType(3.0) * (vin(i, j - 1, k) + vin(i, j, k) + vin(i, j + 1, k));
        _uavg = acrlat1(j) * _uatvpos;
        _vavg = EARTH_RADIUS_RECIP * _vatvpos;
        auto _vres = advectionDriver(vin, i, j, k, _uavg, _vavg, eddlat, eddlon);
        vout(i, j, k) = _vres - tgrlatda1(j) * _uatvpos * _uatvpos;
      }
    }
  }
}


void hadvuv5th_partialfusion(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage1D& acrlat0,
               const Storage1D& acrlat1, const Storage1D& tgrlatda0, const Storage1D& tgrlatda1, Storage3D& uatupos,
               Storage3D& vatupos, Storage3D& uatvpos, Storage3D& vatvpos, Storage3D& uavg, Storage3D& vavg,
               Storage3D& ures, Storage3D& vres, const ElementType eddlat, const ElementType eddlon) {
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _uatupos = (ElementType(1.0) / ElementType(3.0)) * (uin(i - 1, j, k) + uin(i, j, k) + uin(i + 1, j, k));
        auto _vatupos = ElementType(0.25) * (vin(i + 1, j, k) + vin(i + 1, j - 1, k) + vin(i, j, k) + vin(i, j - 1, k));
        auto _uavg = acrlat0(j) * _uatupos;
        auto _vavg = EARTH_RADIUS_RECIP * _vatupos;
        auto _ures = advectionDriver(uin, i, j, k, _uavg, _vavg, eddlat, eddlon);
        uout(i, j, k) = _ures + tgrlatda0(j) * uin(i, j, k) * _vatupos;
      }
    }
  }

  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _uatvpos =
            ElementType(0.25) * (uin(i - 1, j, k) + uin(i, j, k) + uin(i, j + 1, k) + uin(i - 1, j + 1, k));
        auto _vatvpos = ElementType(1.0) / ElementType(3.0) * (vin(i, j - 1, k) + vin(i, j, k) + vin(i, j + 1, k));
        auto _uavg = acrlat1(j) * _uatvpos;
        auto _vavg = EARTH_RADIUS_RECIP * _vatvpos;
        auto _vres = advectionDriver(vin, i, j, k, _uavg, _vavg, eddlat, eddlon);
        vout(i, j, k) = _vres - tgrlatda1(j) * _uatvpos * _uatvpos;
      }
    }
  }
}


void hadvuv5th_openmp(Storage3D& uout, Storage3D& vout, const Storage3D& uin, const Storage3D& vin, const Storage1D& acrlat0,
               const Storage1D& acrlat1, const Storage1D& tgrlatda0, const Storage1D& tgrlatda1, Storage3D& uatupos,
               Storage3D& vatupos, Storage3D& uatvpos, Storage3D& vatvpos, Storage3D& uavg, Storage3D& vavg,
               Storage3D& ures, Storage3D& vres, const ElementType eddlat, const ElementType eddlon) {
  #pragma omp parallel for
  for (int64_t k = 0; k < domain_height; ++k) {
    for (int64_t i = 0; i < domain_size; ++i) {
      for (int64_t j = 0; j < domain_size; ++j) {
        auto _uatupos = (ElementType(1.0) / ElementType(3.0)) * (uin(i - 1, j, k) + uin(i, j, k) + uin(i + 1, j, k));
        auto _vatupos = ElementType(0.25) * (vin(i + 1, j, k) + vin(i + 1, j - 1, k) + vin(i, j, k) + vin(i, j - 1, k));
        auto _uavg = acrlat0(j) * _uatupos;
        auto _vavg = EARTH_RADIUS_RECIP * _vatupos;
        auto _ures = advectionDriver(uin, i, j, k, _uavg, _vavg, eddlat, eddlon);
        uout(i, j, k) = _ures + tgrlatda0(j) * uin(i, j, k) * _vatupos;

        auto _uatvpos = ElementType(0.25) * (uin(i - 1, j, k) + uin(i, j, k) + uin(i, j + 1, k) + uin(i - 1, j + 1, k));
        auto _vatvpos = ElementType(1.0) / ElementType(3.0) * (vin(i, j - 1, k) + vin(i, j, k) + vin(i, j + 1, k));
        _uavg = acrlat1(j) * _uatvpos;
        _vavg = EARTH_RADIUS_RECIP * _vatvpos;
        auto _vres = advectionDriver(vin, i, j, k, _uavg, _vavg, eddlat, eddlon);
        vout(i, j, k) = _vres - tgrlatda1(j) * _uatvpos * _uatvpos;
      }
    }
  }
}

#endif  // HADVUV5TH_H
