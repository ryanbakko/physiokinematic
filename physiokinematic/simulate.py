"""
simulate.py
Simulate a dataset.

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

import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as c

from physiokinematic import utils


def simulate(num, seed=1234):
    """
    Simulate a population of HII regions to test analysis method.

    Inputs:
        num :: integer
            Number of HII regions to generate

    Returns: data
        data :: pandas.DataFrame
            Simulated dataset
    """
    rng = np.random.default_rng(seed)

    # Galactocentric position
    Az = rng.uniform(0.0, 2.0 * np.pi, num)  # rad
    R = 2.0 + np.abs(rng.normal(5.0, 5.0, num))  # kpc
    Xg = -R * np.cos(Az)  # kpc
    Yg = R * np.sin(Az)  # kpc
    Zg = rng.normal(0.0, 0.1, num)  # kpc

    # Heliocentric position
    R0 = utils.__R0
    distance = np.sqrt((Xg + R0) ** 2.0 + Yg**2.0 + Zg**2.0)  # kpc
    glong = np.rad2deg(np.arctan2(Yg, Xg + R0)) % 360.0  # deg
    glat = np.rad2deg(np.arcsin(Zg / distance))  # deg

    # LSR velocity (km/s)
    vlsr = utils.reid19_vlsr(glong, glat, R)
    e_vlsr = 1.0
    vlsr += rng.normal(0.0, e_vlsr, num)

    # Electron temperature (K)
    te = 3900.0 + 250.0 * R
    te += rng.normal(0.0, 500.0, num)
    e_te = 100.0
    te += rng.normal(0.0, e_te, num)

    # Density (cm-3)
    log10_n = rng.normal(1.5, 0.15, num)

    # Ionizing photon rate (s-1)
    log10_q = rng.normal(48.5, 0.75, num)

    # Stromgren radius (pc)
    log10_Rs = log10_q / 3.0 - 2.0 * log10_n / 3.0 - 14.522

    # Angular size (arcsec)
    radius = 206265.0 * (10.0**log10_Rs / (1000.0 * distance))

    # emission measure (pc cm-6)
    log10_em = log10_Rs + 2.0 * log10_n + np.log10(2.0)

    # line width
    line_freq = 8000.0  # MHz
    nonthermal_fwhm = 15.0 * u.km / u.s
    thermal_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(c.k_B * te * u.K / c.m_p)
    fwhm = np.sqrt(nonthermal_fwhm**2.0 + thermal_fwhm**2.0)
    fwhm_kHz = ((fwhm / c.c) * line_freq * u.MHz).to("kHz").value

    # Line opacity and emission
    beam_area = np.pi * 90.0**2.0 / (4.0 * np.log(2.0))  # arcsec2
    tau_line = 1.92e3 * te**-2.5 * 10.0**log10_em / fwhm_kHz
    line = (
        2.0
        * beam_area
        / 206265.0**2.0
        * (c.k_B / c.c**2.0).to("mJy MHz-2 K-1").value
        * line_freq**2.0
        * te
        * (1.0 - np.exp(-tau_line))
    )  # mJy/beam

    # Beam dilution
    source_area = np.pi * radius**2.0 / (4.0 * np.log(2.0))
    beam_dilution = source_area / beam_area
    beam_dilution[beam_dilution > 1.0] = 1.0
    line *= beam_dilution

    # noise
    e_line = 0.1
    line += rng.normal(0.0, e_line, num)

    # missing some electron temperatures
    missing = rng.choice(np.arange(num), int(0.5 * num), replace=False)
    true_te = te.copy()
    te[missing] = np.nan

    # save dataframe
    data = {
        "glong": glong,
        "glat": glat,
        "vlsr": vlsr,
        "e_vlsr": np.ones(num) * e_vlsr,
        "radius": radius,
        "true_te": true_te,
        "te": te,
        "e_te": np.ones(num) * e_te,
        "line": line,
        "e_line": np.ones(num) * e_line,
        "line_unit": ["mJy/beam"] * num,
        "fwhm": fwhm,
        "line_freq": np.ones(num) * line_freq,
        "telescope": ["simulated"] * num,
        "beam_area": np.ones(num) * beam_area,
        "Rgal": R,
        "distance": distance,
        "log10_n": log10_n,
        "log10_q": log10_q,
        "log10_Rs": log10_Rs,
        "log10_em": log10_em,
        "kdar": [""] * num,
    }
    data = pd.DataFrame(data)
    return data
