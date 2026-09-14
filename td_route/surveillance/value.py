"""
Surveillance-mission value (Mission Value Design, slides 8-10).

Each time the satellite images surveillance point I it collects

    V_I = P_I * f( GSD_GM(x(t)) ) * g( eps_I,sun(t) ) * h(t)

and the mission value is the sum over every imaging pass, subject to ONE target
per orbital revolution.

Viewing geometry (roll only, flat Earth)
----------------------------------------
The satellite never pitches: the along-track look angle is 0, so a target is
imaged exactly when it is ABEAM of the satellite (line of sight perpendicular
to the velocity).  At that instant x is the cross-track ground distance from
the sub-satellite point to the target, and (flat Earth, slide 8)

    eta = atan(x / h),   R = h / cos(eta)
    GSD_AT = h beta / cos(eta),   GSD_CT = h beta / cos^2(eta)
    GSD_GM = h beta / cos^(3/2)(eta)

A spherical-Earth variant (theta = eta + x/Re) is kept for comparison.  A pass
counts only if eta <= max_off_nadir_deg.

f -- GIQE-4 (Leachtenauer et al. 1997, Appl. Opt. 36(32)), GSD in INCHES
------------------------------------------------------------------------
    NIIRS = c0 - a log10(GSD) + b log10(RER) - c_h H - c_snr G/SNR
a, b switch on RER >= 0.9.  NIIRS multiplies V_I as-is (not normalised).

g -- sun elevation at the target (slide 10)
--------------------------------------------
    sin eps = sin(phi) sin(delta) + cos(phi) cos(delta) cos(H)
    delta = 23.44 sin(360/365 (n + 284)),   n = day of year (Jan 1 = 1)
    H = 15 (t_LST - 12),   t_LST = t_UTC + lon/15 + EoT/60
    EoT = 9.87 sin 2B - 7.53 cos B - 1.53 sin B,   B = 360/365 (n - 81)
    g = 1/(1+e^{-k(eps-15)}) * 1/(1+e^{-k(60-eps)}),   g = 0 for eps < 0

h -- event window:  h(t) = u(t - t_A) - u(t - t_B)

Orbit
-----
Circular, secular J2: RAAN drifts at -3/2 J2 sqrt(mu) Re^2 a^-7/2 cos i and the
argument of latitude advances at n [1 + 3/2 J2 (Re/a)^2 (4 cos^2 i - 1)].
97.4 deg at 500 km is sun-synchronous, so the initial RAAN is set from the
local time of the ascending node (LTAN) unless raan0_deg is given.
Revolution k runs from one ascending-node crossing to the next.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np

RE_KM = 6378.137          # Earth equatorial radius [km]
MU = 398600.4418          # km^3 / s^2
J2 = 1.08262668e-3
M_PER_INCH = 0.0254
J2000 = np.datetime64("2000-01-01T12:00:00", "us")


# -------------------------------------------------------------------------- #
# Time helpers (UTC, naive numpy datetime64)
# -------------------------------------------------------------------------- #
def to_datetime64(v):
    """ISO string / datetime / datetime64 -> numpy datetime64[us] (UTC).
    None stays None."""
    if v is None:
        return None
    if isinstance(v, np.datetime64):
        return v.astype("datetime64[us]")
    if isinstance(v, _dt.datetime):
        if v.tzinfo is not None:
            v = v.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return np.datetime64(v, "us")
    if isinstance(v, _dt.date):
        return np.datetime64(v, "us")
    s = str(v).strip()
    if s.endswith("Z"):
        s = s[:-1]
    return np.datetime64(s, "us")


def utc_at(epoch, t_s):
    """UTC datetime64 at elapsed seconds t_s from epoch."""
    return epoch + (np.asarray(t_s, dtype=float) * 1e6).astype("timedelta64[us]")


def gmst_rad(utc):
    """Greenwich mean sidereal angle [rad] (UT1 ~ UTC)."""
    d = (utc - J2000) / np.timedelta64(1, "D")
    return np.radians((280.46061837 + 360.98564736629 * d) % 360.0)


# -------------------------------------------------------------------------- #
# f: image quality (GIQE)
# -------------------------------------------------------------------------- #
@dataclass
class GIQE:
    """General Image-Quality Equation, version 4 (GSD_GM in inches)."""

    c0: float = 10.251
    a: tuple = (3.32, 3.16)          # (RER >= threshold, RER < threshold)
    b: tuple = (1.559, 2.817)
    rer_threshold: float = 0.9
    c_snr: float = 0.344
    c_h: float = 0.656
    # GeoEye-1 class EO sensor (Ali et al. 2023)
    rer: float = 0.565
    G: float = 1.0
    snr: float = 17.0
    H: float = 0.749

    @property
    def coeffs(self) -> tuple:
        k = 0 if self.rer >= self.rer_threshold else 1
        return self.a[k], self.b[k]

    def niirs(self, gsd_m):
        """NIIRS for a geometric-mean GSD given in METRES (scalar or array)."""
        a, b = self.coeffs
        gsd_in = np.asarray(gsd_m, dtype=float) / M_PER_INCH
        return (self.c0 - a * np.log10(gsd_in) + b * np.log10(self.rer)
                - self.c_h * self.H - self.c_snr * self.G / self.snr)


@dataclass
class Sensor:
    beta_rad: float = 6.02e-7        # IFOV (GeoEye-1: 0.41 m @ 681 km)
    max_off_nadir_deg: float | None = 60.0   # cross-track roll limit
    earth_model: str = "flat"        # "flat" (slide 8) | "spherical"


def view_geometry(x_km, h_km, earth_model: str = "flat"):
    """Off-nadir eta [rad], slant range R [km], incidence theta [rad] for a
    target a cross-track ground distance x_km from the ground track."""
    x = np.abs(np.asarray(x_km, dtype=float))
    h = np.asarray(h_km, dtype=float)
    if earth_model == "flat":
        eta = np.arctan2(x, h)
        return eta, h / np.cos(eta), eta
    if earth_model == "spherical":
        lam = x / RE_KM
        r = RE_KM + h
        R = np.sqrt(RE_KM**2 + r**2 - 2.0 * RE_KM * r * np.cos(lam))
        eta = np.arcsin(np.clip(RE_KM * np.sin(lam) / R, -1.0, 1.0))
        return eta, R, eta + lam
    raise ValueError("earth_model must be 'flat' or 'spherical', got "
                     + repr(earth_model))


def gsd(x_km, h_km, sensor: Sensor) -> dict:
    """GSD_AT / GSD_CT / GSD_GM [m] plus geometry and a visibility mask."""
    eta, R, theta = view_geometry(x_km, h_km, sensor.earth_model)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = R * 1000.0 * sensor.beta_rad
        ct = at / np.cos(theta)
        gm = np.sqrt(at * ct)
    visible = theta < np.pi / 2.0
    if sensor.max_off_nadir_deg is not None:
        visible &= np.degrees(eta) <= sensor.max_off_nadir_deg
    return {"at": at, "ct": ct, "gm": gm, "eta": eta, "R": R, "theta": theta,
            "visible": visible}


def niirs(x_km, h_km, sensor: Sensor, giqe: GIQE):
    """f: NIIRS achieved imaging a target x_km off-track from altitude h_km.
    NaN where the target is not visible (beyond horizon / roll limit)."""
    g = gsd(x_km, h_km, sensor)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = giqe.niirs(g["gm"])
    return np.where(g["visible"], q, np.nan)


# -------------------------------------------------------------------------- #
# g: sun elevation / illumination
# -------------------------------------------------------------------------- #
@dataclass
class Illumination:
    """g(eps) = 1/(1+exp(-k(eps-eps_low))) * 1/(1+exp(-k(eps_high-eps)))."""
    eps_low_deg: float = 15.0
    eps_high_deg: float = 60.0
    k: float = 0.3
    zero_below_horizon: bool = True  # g = 0 when eps < 0 (night)

    def g(self, eps_deg):
        e = np.asarray(eps_deg, dtype=float)
        val = (1.0 / (1.0 + np.exp(-self.k * (e - self.eps_low_deg)))
               / (1.0 + np.exp(-self.k * (self.eps_high_deg - e))))
        return np.where(e < 0.0, 0.0, val) if self.zero_below_horizon else val


def solar_elevation_deg(lat_deg, lon_deg, utc):
    """Sun elevation at (lat, lon east-positive) for UTC datetime64 (slide 10)."""
    utc = np.asarray(utc, dtype="datetime64[us]")
    day = utc.astype("datetime64[D]")
    n = (day - utc.astype("datetime64[Y]").astype("datetime64[D]")) \
        .astype(int) + 1
    hours = (utc - day) / np.timedelta64(1, "h")
    delta = np.radians(23.44 * np.sin(np.radians(360.0 / 365.0 * (n + 284))))
    B = np.radians(360.0 / 365.0 * (n - 81))
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.53 * np.sin(B)
    t_lst = hours + lon_deg / 15.0 + eot / 60.0
    H = np.radians(15.0 * (t_lst - 12.0))
    phi = np.radians(lat_deg)
    s = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(H)
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


# -------------------------------------------------------------------------- #
# Targets and h(t)
# -------------------------------------------------------------------------- #
@dataclass
class Target:
    name: str
    lat_deg: float
    lon_deg: float                   # east-positive
    priority: float = 1.0            # P_I
    event_start: np.datetime64 | None = None   # t_A (None = open)
    event_end: np.datetime64 | None = None     # t_B (None = open)

    def h(self, utc):
        """Event window h(t) = u(t - t_A) - u(t - t_B)  (1 on [t_A, t_B))."""
        utc = np.asarray(utc, dtype="datetime64[us]")
        on = np.ones(utc.shape, dtype=bool)
        if self.event_start is not None:
            on &= utc >= self.event_start
        if self.event_end is not None:
            on &= utc < self.event_end
        return on.astype(float)

    def eci_unit(self, utc):
        """Unit position vector in the inertial frame (spherical Earth)."""
        lam = np.radians(self.lon_deg) + gmst_rad(utc)
        phi = np.radians(self.lat_deg)
        return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam),
                         np.full(np.shape(lam), np.sin(phi))], axis=-1)


# -------------------------------------------------------------------------- #
# Orbit
# -------------------------------------------------------------------------- #
@dataclass
class CircularOrbit:
    altitude_km: float
    inclination_deg: float
    raan0_deg: float
    arglat0_deg: float = 0.0

    @property
    def a(self) -> float:
        return RE_KM + self.altitude_km

    @property
    def raan_rate(self) -> float:
        """J2 nodal precession [rad/s]."""
        return (-1.5 * J2 * np.sqrt(MU) * RE_KM**2 / self.a**3.5
                * np.cos(np.radians(self.inclination_deg)))

    @property
    def arglat_rate(self) -> float:
        """Mean argument-of-latitude rate with secular J2 [rad/s]."""
        n = np.sqrt(MU / self.a**3)
        ci = np.cos(np.radians(self.inclination_deg))
        return n * (1.0 + 1.5 * J2 * (RE_KM / self.a)**2 * (4.0 * ci**2 - 1.0))

    @property
    def period_s(self) -> float:
        return 2.0 * np.pi / self.arglat_rate

    def frame(self, t_s):
        """Unit radial r, velocity v and orbit-normal w (= r x v, LEFT of the
        velocity) in the inertial frame at elapsed seconds t_s."""
        t = np.asarray(t_s, dtype=float)
        O = np.radians(self.raan0_deg) + self.raan_rate * t
        u = np.radians(self.arglat0_deg) + self.arglat_rate * t
        i = np.radians(self.inclination_deg)
        P = np.stack([np.cos(O), np.sin(O), np.zeros_like(O)], axis=-1)
        Q = np.stack([-np.cos(i) * np.sin(O), np.cos(i) * np.cos(O),
                      np.full(O.shape, np.sin(i))], axis=-1)
        cu, su = np.cos(u)[..., None], np.sin(u)[..., None]
        r = cu * P + su * Q
        v = -su * P + cu * Q
        w = np.stack([np.sin(i) * np.sin(O), -np.sin(i) * np.cos(O),
                      np.full(O.shape, np.cos(i))], axis=-1)
        return r, v, w

    def revolution(self, t_s):
        """Revolution index: +1 at every ascending-node crossing (rev 0
        contains the epoch)."""
        u = np.radians(self.arglat0_deg) + self.arglat_rate * np.asarray(t_s)
        return np.floor(u / (2.0 * np.pi)).astype(int) \
            - int(np.floor(np.radians(self.arglat0_deg) / (2.0 * np.pi)))


def raan_from_ltan(epoch, ltan_hours: float) -> float:
    """RAAN [deg] putting the ascending node at local MEAN solar time
    ltan_hours at the epoch:  Omega = GMST + 15 (LTAN - t_UTC)."""
    hours = (epoch - epoch.astype("datetime64[D]")) / np.timedelta64(1, "h")
    return float((np.degrees(gmst_rad(epoch)) + 15.0 * (ltan_hours - hours))
                 % 360.0)


# -------------------------------------------------------------------------- #
# Mission
# -------------------------------------------------------------------------- #
@dataclass
class SurveillanceMission:
    targets: list
    sensor: Sensor = field(default_factory=Sensor)
    giqe: GIQE = field(default_factory=GIQE)
    illumination: Illumination = field(default_factory=Illumination)
    altitude_km: float = 500.0
    inclination_deg: float = 97.4
    ltan_hours: float = 22.5         # 22:30 ascending = 10:30 descending
    raan0_deg: float | None = None   # overrides ltan_hours when given
    arglat0_deg: float = 0.0
    wet_mass_kg: float = 5000.0
    fuel_mass_kg: float = 1500.0
    thrust_N: float = 490.0
    isp_s: float = 315.5
    epoch_utc: np.datetime64 = field(
        default_factory=lambda: np.datetime64("2027-04-01T00:00:00", "us"))
    duration_days: float = 30.0
    search_step_s: float = 60.0      # pass-search sampling step

    def orbit(self) -> CircularOrbit:
        raan = (self.raan0_deg if self.raan0_deg is not None
                else raan_from_ltan(self.epoch_utc, self.ltan_hours))
        return CircularOrbit(self.altitude_km, self.inclination_deg,
                             raan, self.arglat0_deg)

    # ------------------------------------------------------------------ #
    def find_passes(self) -> list:
        """Every instant a target is ABEAM (along-track look angle 0) on the
        near side, with its geometry and value terms.  Passes beyond the roll
        limit are dropped; night / out-of-event passes are kept with V = 0."""
        orb = self.orbit()
        T = np.arange(0.0, self.duration_days * 86400.0 + self.search_step_s,
                      self.search_step_s)
        h = self.altitude_km
        passes = []
        for tgt in self.targets:
            def along(t):
                rT = tgt.eci_unit(utc_at(self.epoch_utc, t))
                r, v, _ = orb.frame(t)
                return np.sum(rT * v, axis=-1), np.sum(rT * r, axis=-1)

            a, c = along(T)
            # target moves from ahead (+) to behind (-) on the visible side
            k = np.nonzero((a[:-1] > 0.0) & (a[1:] <= 0.0) & (c[:-1] > 0.0))[0]
            lo, hi = T[k], T[k + 1]
            for _ in range(40):                       # vectorised bisection
                mid = 0.5 * (lo + hi)
                ahead = along(mid)[0] > 0.0
                lo, hi = np.where(ahead, mid, lo), np.where(ahead, hi, mid)
            tp = 0.5 * (lo + hi)
            if tp.size == 0:
                continue

            utc = utc_at(self.epoch_utc, tp)
            rT = tgt.eci_unit(utc)
            r, v, w = orb.frame(tp)
            x = RE_KM * np.arctan2(np.sum(rT * w, -1), np.sum(rT * r, -1))
            geo = gsd(x, h, self.sensor)
            q = niirs(x, h, self.sensor, self.giqe)
            eps = solar_elevation_deg(tgt.lat_deg, tgt.lon_deg, utc)
            gval = self.illumination.g(eps)
            hval = tgt.h(utc)
            rev = orb.revolution(tp)
            for m in np.nonzero(geo["visible"])[0]:
                V = tgt.priority * q[m] * gval[m] * hval[m]
                passes.append({
                    "target": tgt.name, "t_s": float(tp[m]),
                    "day": float(tp[m] / 86400.0),
                    "utc": str(utc[m].astype("datetime64[s]")) + "Z",
                    "rev": int(rev[m]),
                    "node": "asc" if v[m, 2] > 0.0 else "desc",
                    "x_km": float(x[m]),
                    "eta_deg": float(np.degrees(geo["eta"][m])),
                    "gsd_gm_m": float(geo["gm"][m]), "niirs": float(q[m]),
                    "eps_sun_deg": float(eps[m]), "g": float(gval[m]),
                    "h": float(hval[m]), "P": tgt.priority, "V": float(V),
                    "selected": False})
        passes.sort(key=lambda p: p["t_s"])
        return passes


def select_per_revolution(passes: list) -> list:
    """One target per revolution.  On a fixed (unmanoeuvred) orbit the choice
    in one revolution does not affect any other, so taking the highest-V pass
    in every revolution maximises the total exactly.  Marks `selected`."""
    best = {}
    for p in passes:
        if p["V"] > 0.0 and (p["rev"] not in best or p["V"] > best[p["rev"]]["V"]):
            best[p["rev"]] = p
    for p in best.values():
        p["selected"] = True
    return sorted(best.values(), key=lambda p: p["t_s"])


def from_config(cfg: dict) -> SurveillanceMission:
    """Build a SurveillanceMission from the `surveillance` config section."""
    s = cfg.get("surveillance", {})
    orb, sc = s.get("orbit", {}), s.get("spacecraft", {})
    sen, gq, il = s.get("sensor", {}), s.get("giqe", {}), s.get("illumination", {})
    val = s.get("value", {})

    targets = []
    for k, t in enumerate(s.get("targets", []), start=1):
        ev = t.get("event") or [None, None]
        targets.append(Target(
            name=str(t.get("name", f"T{k}")),
            lat_deg=float(t["lat"]), lon_deg=float(t["lon"]),
            priority=float(t.get("priority", 1.0)),
            event_start=to_datetime64(ev[0]), event_end=to_datetime64(ev[1])))

    giqe = GIQE()
    for key in ("c0", "rer_threshold", "c_snr", "c_h", "rer", "G", "snr", "H"):
        if key in gq:
            setattr(giqe, key, float(gq[key]))
    for key in ("a", "b"):
        if key in gq:
            setattr(giqe, key, tuple(float(v) for v in gq[key]))

    mx = sen.get("max_off_nadir_deg", 60.0)
    return SurveillanceMission(
        targets=targets,
        sensor=Sensor(beta_rad=float(sen.get("beta_rad", 6.02e-7)),
                      max_off_nadir_deg=None if mx is None else float(mx),
                      earth_model=sen.get("earth_model", "flat")),
        giqe=giqe,
        illumination=Illumination(
            eps_low_deg=float(il.get("eps_low_deg", 15.0)),
            eps_high_deg=float(il.get("eps_high_deg", 60.0)),
            k=float(il.get("k", 0.3)),
            zero_below_horizon=bool(il.get("zero_below_horizon", True))),
        altitude_km=float(orb.get("altitude_km", 500.0)),
        inclination_deg=float(orb.get("inclination_deg", 97.4)),
        ltan_hours=float(orb.get("ltan_hours", 22.5)),
        raan0_deg=(None if orb.get("raan0_deg") is None
                   else float(orb["raan0_deg"])),
        arglat0_deg=float(orb.get("arglat0_deg") or 0.0),
        wet_mass_kg=float(sc.get("wet_mass_kg", 5000.0)),
        fuel_mass_kg=float(sc.get("fuel_mass_kg", 1500.0)),
        thrust_N=float(sc.get("thrust_N", 490.0)),
        isp_s=float(sc.get("isp_s", 315.5)),
        epoch_utc=to_datetime64(s.get("epoch_utc") or "2027-04-01T00:00:00Z"),
        duration_days=float(s.get("duration_days") or 30.0),
        search_step_s=float(val.get("search_step_s", 60.0)))
