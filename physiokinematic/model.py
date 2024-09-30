"""
model.py
Forward model HII region distances and physical properties.

Copyright(C) 2023 by
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

TODO: can we perform a similar analysis on HII region candidates since they
are constrained by the size?
"""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import astropy.constants as c

from physiokinematic import utils, simulate, loader


def model(data):
    """
    Generates an instance of the forward model applied to a given dataset.

    Inputs:
        data :: pd.DataFrame
            Dataset
    """
    # line width in kHz
    data = data.assign(fwhm_kHz=1000.0 * data["line_freq"] * data["fwhm"] / c.c.to("km/s").value)

    with pm.Model(coords={"data": data.index, "kdar": ["N", "F"]}) as model:
        # Galactocentric radius
        Rmin = utils.__R0 * np.abs(np.sin(np.deg2rad(data["glong"])))
        Rmin = pt.switch(pt.lt(np.cos(np.deg2rad(data["glong"])), 0.0), utils.__R0, Rmin)
        Rgal_offset = pm.HalfNormal("Rgal_offset", sigma=10.0, dims="data")
        Rgal = pm.Deterministic("Rgal", Rmin + Rgal_offset, dims="data")

        # LSR velocity (km/s)
        vlsr = utils.reid19_vlsr(data["glong"].to_numpy(), data["glat"].to_numpy(), Rgal)
        _ = pm.Normal(
            "vlsr",
            mu=vlsr,
            sigma=data["e_vlsr"].to_numpy(),
            observed=data["vlsr"].to_numpy(),
            dims="data",
        )

        # Electron temperature (K)
        log10_te = pm.Normal("log10_te", mu=3.5, sigma=0.5, dims="data")
        te = 10.0**log10_te
        _ = pm.Normal("te", mu=te, sigma=data["e_te"].to_numpy(), observed=data["te"].to_numpy(), dims="data")

        # distance (kpc)
        distance_near, distance_far = utils.distance(data["glong"].to_numpy(), data["glat"].to_numpy(), Rgal)
        distance = pm.Deterministic(
            "distance",
            pt.stack([distance_near, distance_far], axis=-1),
            dims=["data", "kdar"],
        )

        # KDAR
        kdar_w = pm.Dirichlet("kdar_w", a=np.ones((len(data), 2)), dims=["data", "kdar"])

        # Ionizing photon rate
        log10_q = pm.Normal("log10_q", mu=48.5, sigma=0.75, dims=["data", "kdar"])

        # Electron density
        log10_n = pm.Normal("log10_n", mu=1.5, sigma=0.15, dims=["data", "kdar"])

        # Stromgren radius
        log10_Rs = pm.Deterministic(
            "log10_Rs",
            log10_q / 3.0 - 2.0 * log10_n / 3.0 - 14.522,
            dims=["data", "kdar"],
        )

        # Radius
        radius_mu = pm.Deterministic(
            "radius_mu",
            206265.0 * (10.0**log10_Rs / (1000.0 * distance)),
            dims=["data", "kdar"],
        )
        radius = pm.NormalMixture(
            "radius",
            mu=radius_mu,
            sigma=1.0,
            w=kdar_w,
            observed=data["radius"].to_numpy(),
            dims="data",
        )

        # Emission measure
        log10_em = pm.Deterministic(
            "log10_em",
            log10_Rs + 2.0 * log10_n + np.log10(2.0),
            dims=["data", "kdar"],
        )

        # line brightness
        tau_line = pm.Deterministic(
            "tau_line",
            1.92e3 * te[:, None] ** -2.5 * 10.0**log10_em / data["fwhm_kHz"].to_numpy()[:, None],
            dims=["data", "kdar"],
        )
        line_mu = (
            2.0
            * data["beam_area"].to_numpy()[:, None]
            / 206265.0**2.0
            * (c.k_B / c.c**2.0).to("mJy MHz-2 K-1").value
            * data["line_freq"].to_numpy()[:, None] ** 2.0
            * te[:, None]
            * (1.0 - np.exp(-tau_line))
        )  # mJy/beam

        # beam dilution
        source_area = np.pi * radius**2.0 / (4.0 * np.log(2.0))
        beam_dilution = source_area / data["beam_area"].to_numpy()
        beam_dilution = pt.clip(beam_dilution, 0.0, 1.0)
        line_mu = pm.Deterministic(
            "line_mu",
            line_mu * beam_dilution[:, None],
            dims=["data", "kdar"],
        )
        _ = pm.NormalMixture(
            "line",
            mu=line_mu,
            sigma=data["e_line"].to_numpy()[:, None],
            w=kdar_w,
            observed=data["line"].to_numpy(),
            dims="data",
        )
    return model
