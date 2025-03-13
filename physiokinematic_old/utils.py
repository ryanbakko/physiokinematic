"""
utils.py
Utilities for analysis.

Copyright(C) 2023-2024 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

# Reid+2019 A5 rotation model parameters
__R0 = 8.166  # kpc
__Usun = 10.449  # km/s
__Vsun = 12.092  # km/s
__Wsun = 7.729  # km/s
__a2 = 0.977
__a3 = 1.623

# IAU defined LSR
__Ustd = 10.27  # km/s
__Vstd = 15.32  # km/s
__Wstd = 7.74  # km/s


def reid19_theta(R, R0=__R0, a2=__a2, a3=__a3):
    """
    Calculate the Reid et al. (2019) circular rotation speed at a
    given Galactocentric radius.

    Inputs:
        R :: scalar (kpc)
            Galactocentric radius
        R0 :: scalar (kpc)
            Solar Galactocentric radius
        a2, a3 :: scalar
            Parameters that define rotation curve

    Returns: theta
        theta :: scalar (km/s)
            Circular rotation speed
    """
    rho = R / (a2 * R0)
    lam = (a3 / 1.5) ** 5.0
    loglam = np.log10(lam)
    term1 = 200.0 * lam**0.41
    term2 = np.sqrt(
        0.8 + 0.49 * loglam + 0.75 * np.exp(-0.4 * lam) / (0.47 + 2.25 * lam**0.4)
    )
    term3 = (0.72 + 0.44 * loglam) * 1.97 * rho**1.22 / (rho**2.0 + 0.61) ** 1.43
    term4 = 1.6 * np.exp(-0.4 * lam) * rho**2.0 / (rho**2.0 + 2.25 * lam**0.4)
    theta = term1 / term2 * np.sqrt(term3 + term4)
    return theta


def reid19_vlsr(
    glong,
    glat,
    R,
    R0=__R0,
    a2=__a2,
    a3=__a3,
    Usun=__Usun,
    Vsun=__Vsun,
    Wsun=__Wsun,
):
    """
    Calculate the Reid et al. (2019) rotation curve LSR velocity
    at a given position.

    Inputs:
        glong, glat :: scalars (deg)
            Galactic longitude and latitude
        R :: scalar (kpc)
            Galactocentric radius
        R0 :: scalar (kpc)
            Solar Galactocentric radius
        a2, a3 :: scalar
            Parameters that define rotation curve
        Usun, Vsun, Wsun :: scalars (km/s)
            Solar motion relative to the LSR

    Returns: vlsr
        vlsr :: scalar (km/s)
            LSR velocity
    """
    # Circular velocities
    theta = reid19_theta(R, R0=R0, a2=a2, a3=a3)
    theta0 = reid19_theta(R0, R0=R0, a2=a2, a3=a3)

    # Radial velocity relative to LSR
    sin_glong = np.sin(np.deg2rad(glong))
    cos_glat = np.cos(np.deg2rad(glat))
    vlsr = R0 * sin_glong * cos_glat * (theta / R - theta0 / R0)

    # Difference between solar motion and IAU definition
    U = (__Ustd - Usun) * np.cos(np.deg2rad(glong))
    V = (__Vstd - Vsun) * sin_glong
    W = (__Wstd - Wsun) * np.sin(np.deg2rad(glat))
    return vlsr + (U + V) * cos_glat + W


def distance(glong, glat, Rgal, R0=__R0):
    """
    Return the distances at which a given line-of-sight crosses a
    given Galactocentric radius.

    Inputs:
        glong, glat :: scalars (deg)
            Galactic longitude and latitude
        Rgal :: scalar (kpc)
            Galactocentric radius
        R0 :: scalar (kpc)
            Solar Galactocentric radius

    Returns: [d_near, d_far]
        d_near, d_far :: scalars (kpc)
            Near and far distances. Near distance is negative when not allowed.
            Both distances are nan if line of sight never crosses Rgal.
    """
    glong_rad = np.deg2rad(glong)
    cos_glat = np.cos(np.deg2rad(glat))

    # tangent point distance and Rgal
    Rtp = R0 * np.abs(np.sin(glong_rad))
    dtp = R0 * np.cos(glong_rad)

    # midplane distance
    d_near = dtp - np.sqrt(Rgal**2.0 - Rtp**2.0)
    d_far = dtp + np.sqrt(Rgal**2.0 - Rtp**2.0)

    return d_near / cos_glat, d_far / cos_glat
