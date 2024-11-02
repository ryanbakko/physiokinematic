"""
NEWmodel.py
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
            Dataset4
    """
    # line width in kHz
    data = data.assign(
        fwhm_kHz=1000.0 * data["line_freq"] * data["fwhm"] / c.c.to("km/s").value
    )

    with pm.Model(coords={"data": data.index}) as model:
        distance = pm.HalfNormal("distance", sigma=5.0, dims="data")

        # Galactocentric radius
        Rgal = pm.Deterministic(
            "Rgal",
            pt.sqrt(
                utils.__R0**2
                + distance**2
                - 2 * utils.__R0 * distance * np.cos(np.deg2rad(data["glong"]))
            ),
            dims="data",
        )

        # LSR velocity (km/s)
        vlsr = utils.reid19_vlsr(
            data["glong"].to_numpy(), data["glat"].to_numpy(), Rgal
        )
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
        _ = pm.Normal(
            "te",
            mu=te,
            sigma=data["e_te"].to_numpy(),
            observed=data["te"].to_numpy(),
            dims="data",
        )

        # Ionizing photon rate
        log10_q = pm.Normal("log10_q", mu=48.5, sigma=0.75, dims=["data"])

        # Electron density
        log10_n = pm.Normal("log10_n", mu=1.5, sigma=0.15, dims=["data"])

        # Stromgren radius
        log10_Rs = pm.Deterministic(
            "log10_Rs",
            log10_q / 3.0 - 2.0 * log10_n / 3.0 - 14.522,
            dims=["data"],
        )

        # Radius
        radius_mu = pm.Deterministic(
            "radius_mu",
            206265.0 * (10.0**log10_Rs / (1000.0 * distance)),
            dims=["data"],
        )
        radius = pm.Normal(
            "radius",
            mu=radius_mu,
            sigma=1.0,
            observed=data["radius"].to_numpy(),
            dims="data",
        )

        # Emission measure
        log10_em = pm.Deterministic(
            "log10_em",
            log10_Rs + 2.0 * log10_n + np.log10(2.0),
            dims=["data"],
        )

        # line brightness
        tau_line = pm.Deterministic(
            "tau_line",
            1.92e3
            * te[:, None] ** -2.5
            * 10.0**log10_em
            / data["fwhm_kHz"].to_numpy()[:, None],
            dims=["data"],
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
            line_mu * beam_dilution,
            dims=["data"],
        )
        _ = pm.NormalMixture(
            "line",
            mu=line_mu,
            sigma=data["e_line"].to_numpy(),
            observed=data["line"].to_numpy(),
            dims="data",
        )
    return model


# 11/2/24 additions
import pandas as pd
import matplotlib.pyplot as plt


def simulate_hii_regions(num_regions=100, seed=42):
    """
    Simulates HII regions with random data.

    Parameters:
        num_regions (int): Number of HII regions to simulate.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing simulated HII region data.
    """
    np.random.seed(seed)  # For reproducibility
    gal_longitude = np.random.uniform(
        0, 360, num_regions
    )  # Galactic longitude in degrees
    gal_latitude = np.random.uniform(
        -90, 90, num_regions
    )  # Galactic latitude in degrees
    velocity = np.random.uniform(-250, 250, num_regions)  # Velocity in km/s
    apparent_size = np.random.uniform(1, 20, num_regions)  # Apparent size in arcmin
    absolute_size = np.random.uniform(10, 100, num_regions)  # Absolute size in parsecs

    # Creating a DataFrame to hold the simulated data
    hii_data = pd.DataFrame(
        {
            "Gal_Longitude": gal_longitude,
            "Gal_Latitude": gal_latitude,
            "Velocity": velocity,
            "Apparent_Size": apparent_size,
            "Absolute_Size": absolute_size,
        }
    )

    return hii_data


def plot_galaxy_map(hii_data):
    """
    Plots a map of the simulated HII regions in the galaxy.

    Parameters:
        hii_data (pd.DataFrame): DataFrame containing HII region data.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        hii_data["Gal_Longitude"],
        hii_data["Gal_Latitude"],
        s=hii_data["Apparent_Size"],
        alpha=0.7,
        marker="o",
    )
    plt.xlabel("Galactic Longitude (degrees)")
    plt.ylabel("Galactic Latitude (degrees)")
    plt.title(
        "Map of the Galaxy with Simulated HII Regions (Apparent Size Proportional)"
    )
    plt.grid(True)
    plt.show()


# Usage example
hii_data = simulate_hii_regions(num_regions=100)  # Step 1: Simulate HII regions
plot_galaxy_map(hii_data)  # Step 2: Plot the galaxy map


def plot_longitude_velocity_diagram(hii_data):
    """
    Plots a Longitude-Velocity (L-V) diagram of the simulated HII regions.

    Parameters:
        hii_data (pd.DataFrame): DataFrame containing HII region data.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        hii_data["Gal_Longitude"],
        hii_data["Velocity"],
        s=hii_data["Absolute_Size"],
        alpha=0.7,
        marker="o",
    )
    plt.xlabel("Galactic Longitude (degrees)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Longitude-Velocity Diagram (Absolute Size Proportional)")
    plt.grid(True)
    plt.show()


# Usage example
plot_longitude_velocity_diagram(hii_data)  # Step 3: Plot the Longitude-Velocity diagram


def simulate_te_vs_rgal(
    num_regions=100, te_slope=50, te_intercept=1000, noise_level=100, seed=42
):
    """
    Simulates a positive linear relationship between electron temperature (Te) and Galactic radius (Rgal).

    Parameters:
        num_regions (int): Number of HII regions to simulate.
        te_slope (float): Slope for the Te vs Rgal linear relationship.
        te_intercept (float): Intercept for the Te vs Rgal relationship.
        noise_level (float): Standard deviation of Gaussian noise to add to the Te values.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing simulated Rgal and Te data.
    """
    np.random.seed(seed)  # For reproducibility
    rgal = np.random.uniform(0, 15, num_regions)  # Simulated Galactic radius in kpc
    te = (
        te_slope * rgal + te_intercept + np.random.normal(0, noise_level, num_regions)
    )  # Linear Te with noise

    # Creating a DataFrame to hold the simulated data
    te_rgal_data = pd.DataFrame({"Rgal": rgal, "Te": te})

    return te_rgal_data


def plot_te_vs_rgal(te_rgal_data):
    """
    Plots a Te vs. Rgal figure showing a positive linear relationship.

    Parameters:
        te_rgal_data (pd.DataFrame): DataFrame containing Te and Rgal data.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(te_rgal_data["Rgal"], te_rgal_data["Te"], alpha=0.7, marker="o")
    plt.xlabel("Galactic Radius (Rgal) [kpc]")
    plt.ylabel("Electron Temperature (Te) [K]")
    plt.title("Te vs. Rgal: Positive Linear Relationship")
    plt.grid(True)
    plt.show()


# Usage example
te_rgal_data = simulate_te_vs_rgal(num_regions=100)  # Simulate Te vs Rgal data
plot_te_vs_rgal(te_rgal_data)  # Step 5: Plot the Te vs Rgal figure
