"""
model.py
Forward model HII region distance and physical properties.

Copyright(C) 2023-2025 by
Trey V. Wenger; tvwenger@gmail.com
Ryan Bakko

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

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import astropy.constants as c

from physiokinematic import utils


def model(data):
    """
    Generates an instance of the forward model applied to a given data point.

    Inputs:
        data :: pd.Series
            Data for a single HII region
    """
    # Quadrant 3 or 4
    is_q3_q4 = np.cos(np.deg2rad(data["glong"])) < 0.0

    # Outer galaxy
    sin_glong = np.sin(np.deg2rad(data["glong"]))
    is_outer_galaxy = is_q3_q4 or np.sign(sin_glong) != np.sign(data["vlsr"])
    kdar = ["F"] if is_outer_galaxy else ["F", "N"]

    # line width in kHz
    data["fwhm_kHz"] = 1000.0 * data["line_freq"] * data["fwhm"] / c.c.to("km/s").value

    with pm.Model(coords={"kdar": kdar}) as model:
        # Kinematic distance ambiguity
        if is_outer_galaxy:
            kdar_w = np.array([1.0])
        else:
            kdar_w = pm.Dirichlet("kdar_w", a=[0.5, 0.5], dims="kdar")

        # Galactocentric radius (kpc)
        Rsigma = 10.0
        Rmin = utils.__R0 * np.abs(np.sin(np.deg2rad(data["glong"])))
        Rmin_far = utils.__R0 if is_q3_q4 else Rmin
        Rmax_far = 35.0  # kpc
        if is_outer_galaxy:
            Rgal = pm.TruncatedNormal(
                "Rgal", mu=0.0, sigma=Rsigma, lower=Rmin_far, upper=Rmax_far, dims="kdar"
            )
        else:
            Rmin_near = Rmin
            Rmax_near = utils.__R0
            Rgal = pm.TruncatedNormal(
                "Rgal",
                mu=0.0,
                sigma=Rsigma,
                lower=[Rmin_far, Rmin_near],
                upper=[Rmax_far, Rmax_near],
                dims="kdar",
            )

        # Distance (kpc)
        Rtan = utils.__R0 * np.cos(np.deg2rad(data["glong"]))
        Rdiff = np.sqrt(Rgal**2 - Rmin**2)
        d_far = Rtan + Rdiff[0]
        if is_outer_galaxy:
            distance = pm.Deterministic("distance", pt.stack([d_far]), dims="kdar")
        else:
            d_near = Rtan - Rdiff[1]
            distance = pm.Deterministic(
                "distance", pt.stack([d_far, d_near]), dims="kdar"
            )

        # LSR velocity (km/s)
        vlsr = utils.reid19_vlsr(data["glong"], data["glat"], Rgal)
        _ = pm.NormalMixture(
            "vlsr",
            mu=vlsr,
            sigma=data["e_vlsr"],
            w=kdar_w,
            observed=data["vlsr"],
        )

        # Electron temperature (K)
        log10_te = pm.Normal("log10_te", mu=3.5, sigma=0.5)
        te = 10.0**log10_te
        if not np.isnan(data["te"]):
            _ = pm.Normal(
                "te",
                mu=te,
                sigma=data["e_te"],
                observed=data["te"],
            )

        # Emission measure (pc cm-6)
        log10_em = pm.Normal("log10_em", mu=4.0, sigma=1.0, dims="kdar")

        # Stromgren radius (pc)
        log10_Rs = pm.Deterministic(
            "log10_Rs",
            pt.log10(data["radius"] * 1000.0 * distance / 206265.0),
            dims="kdar",
        )

        # Electron density (cm-3)
        log10_n = pm.Deterministic(
            "log10_n",
            0.5 * log10_em - 0.5 * log10_Rs - 0.5 * np.log10(2.0),
            dims="kdar",
        )

        # Ionizing photon rate (s-1)
        _ = pm.Deterministic(
            "log10_q", 3.0 * log10_Rs + 2.0 * log10_n + 3.0 * 14.522, dims="kdar"
        )

        # RRL optical depth
        log10_tau_line = pm.Deterministic(
            "log10_tau_line",
            3.2833 - 2.5 * log10_te + log10_em - np.log10(data["fwhm_kHz"]),
            dims="kdar",
        )

        # Undiluted RRL brightness (mJy/beam)
        line_mu = (
            2.0
            * data["beam_area"]
            / 206265.0**2.0
            * (c.k_B / c.c**2.0).to("mJy MHz-2 K-1").value
            * data["line_freq"] ** 2.0
            * te
            * (1.0 - np.exp(-(10.0**log10_tau_line)))
        )

        # Beam dilution
        source_area = np.pi * data["radius"] ** 2.0 / (4.0 * np.log(2.0))
        beam_dilution = source_area / data["beam_area"]
        beam_dilution = pt.clip(beam_dilution, 0.0, 1.0)

        # Diluted RRL brightness (mJy/beam)
        line_mu = pm.Deterministic(
            "line_mu",
            line_mu * beam_dilution,
            dims="kdar",
        )
        _ = pm.NormalMixture(
            "line",
            mu=line_mu,
            sigma=data["e_line"],
            w=kdar_w,
            observed=data["line"],
        )
    return model
