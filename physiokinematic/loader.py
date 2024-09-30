"""
loader.py
Load HII region data from database.

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

import sqlite3
import pandas as pd
import numpy as np


def loader(db):
    """
    Returns HII region data as a pandas DataFrame.

    Inputs:
        db :: string
            Path to database file

    Returns:
        data :: pd.DataFrame
            HII region data
    """
    with sqlite3.connect(db) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.execute(
            """
        SELECT cat.gname, cat.glong, cat.glat, cat.radius,
        det.vlsr, det.e_vlsr, det.line, det.e_line, det.line_unit, det.beam_area, det.line_freq,
        det.fwhm, det.e_fwhm, det.te, det.e_te,
        det.source, det.author, det.telescope, cat.kdar,
        dist.Rgal, dist.near, dist.far, dist.tangent,
        dist.vlsr_tangent, par.plx, par.author as dist_author,
        MAX(CASE WHEN det.te IS NULL THEN 1 ELSE 0 END) AS te_priority
        FROM Detections det
        INNER JOIN CatalogDetections catdet ON catdet.detection_id = det.id
        INNER JOIN Catalog cat ON catdet.catalog_id = cat.id
        LEFT JOIN Distances_Reid2019 dist ON dist.catalog_id = cat.id
        LEFT JOIN CatalogParallax catpar ON catpar.catalog_id = cat.id
        LEFT JOIN Parallax par ON catpar.parallax_id = par.id
        WHERE det.vlsr IS NOT NULL AND det.line IS NOT NULL
        AND det.type = "peak" AND det.component IS NULL
        GROUP BY cat.gname, det.component HAVING MIN(det.e_vlsr)
        """
        )
        data = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    bad_te = data["te"] == ""
    data.loc[bad_te, "te"] = np.nan
    data = data.assign(te=data["te"].astype(float))

    # convert brightness temperature (mK) to brightness (mJy/beam)
    telescopes = ["GBT", "NRAO 140'", "NRAO 140 Foot", "Arecibo", "Parkes"]
    gains = [0.5, 3.85, 3.85, 0.2, 1.3]  # Jy/K
    for telescope, gain in zip(telescopes, gains):
        fix = (data["telescope"] == telescope) & (data["line_unit"] == "mK")
        data.loc[fix, "line"] *= gain
        data.loc[fix, "e_line"] *= gain
        data.loc[fix, "line_unit"] = "mJy/beam"

    # scale line brightness to 8 GHz assuming S \propto \nu
    data["line"] *= (8000.0 / data["line_freq"]) ** 1.0
    data["e_line"] *= (8000.0 / data["line_freq"]) ** 1.0
    data["line_freq"] = 8000.0

    # scale single-dish electron temperatures by 90% (see Wenger et al. 2019)
    is_sd = (data["telescope"] != "ATCA") & (data["telescope"] != "JVLA")
    data.loc[is_sd, "te"] *= 0.9
    data.loc[is_sd, "e_te"] *= 0.9

    # drop the crazy GC and other sources
    drop = (data["glong"] > 354.0) & (np.abs(data["vlsr"]) > 50.0)
    drop |= (data["glong"] < 5.0) & (data["vlsr"] > 50.0)
    drop |= (data["glong"] > 300.0) & (np.abs(data["vlsr"]) > 150.0)
    drop |= (data["glong"] > 295.0) & (data["glong"] < 320.0) & (data["vlsr"] < -100.0)
    drop |= (data["te"] > 11000.0) | (data["te"] < 3000.0)
    data = data[~drop].reset_index(drop=True)

    # catch missing uncertainties
    data.loc[data["e_vlsr"] <= 0.1, "e_vlsr"] = 0.1
    data.loc[data["e_te"].isna(), "e_te"] = 500.0

    return data
