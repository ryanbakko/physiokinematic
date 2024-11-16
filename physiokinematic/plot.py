import pymc as pm
import pytensor.tensor as pt
import numpy as np
import astropy.constants as c
import matplotlib.pyplot as plt


def plot_longitude_velocity_diagram(hii_data):
    """
    Plots a Longitude-Velocity (L-V) diagram of the simulated HII regions.

    Parameters:
        hii_data (pd.DataFrame): DataFrame containing HII region data.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        hii_data["glong"],
        hii_data["vlsr"],
        s=hii_data["radius"],
        alpha=0.7,
        marker="o",
    )
    plt.xlabel("Galactic Longitude (degrees)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Longitude-Velocity Diagram (Apparent Size Proportional)")
    plt.grid(True)
    plt.show()


def plot_te_vs_rgal(te_rgal_data):
    """
    Plots a Te vs. Rgal figure showing a positive linear relationship.

    Parameters:
        te_rgal_data (pd.DataFrame): DataFrame containing Te and Rgal data.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(te_rgal_data["Rgal"], te_rgal_data["te"], alpha=0.7, marker="o")
    plt.xlabel("Galactic Radius (Rgal) [kpc]")
    plt.ylabel("Electron Temperature (Te) [K]")
    plt.title("Te vs. Rgal: Positive Linear Relationship")
    plt.grid(True)
    plt.show()


def plot_galaxy_map(hii_data):
    """
    Plots a map of the simulated HII regions in the galaxy.

    Parameters:
        hii_data (pd.DataFrame): DataFrame containing HII region data.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        hii_data["glong"],
        hii_data["glat"],
        s=hii_data["radius"],
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
