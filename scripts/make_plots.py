from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface.full as si

y_lims = [
    [1980, 2680],
]

chunk_filepath = Path(
    "/Users/christopherhalcrow/Work/Edinburgh/Harry_Project/fast_curate_demo/analyzers/aeon_Raw/NeuropixelsV2_ProbeB_2.arrow"
)
processed_folder = Path("processed")

chunk_filepaths = [chunk_filepath, chunk_filepath, chunk_filepath]

num_files = len(chunk_filepaths)

unit_fig, unit_axs = plt.subplots(ncols=num_files, nrows=4, figsize=(3 + num_files, 10))
amp_fig, amp_axs = plt.subplots(
    ncols=num_files, nrows=4, figsize=(3 + 2 * num_files, 10)
)

for chunk_id, chunk in enumerate(chunk_filepaths):
    for shank_id in [0, 1, 2, 3]:
        unit_ax = unit_axs[shank_id][chunk_id]
        amp_ax = amp_axs[shank_id][chunk_id]

        analyzer_output_folder = processed_folder / (
            chunk_filepath.name.split(".")[0] + f"_lupin_analyzer_s{shank_id}.zarr"
        )
        analyzer = si.load_sorting_analyzer(analyzer_output_folder)

        si.plot_drift_raster_map(
            sorting_analyzer=analyzer, ax=amp_ax, scatter_decimate=10
        )
        amp_ax.set_ylim(y_lims[0][0], y_lims[0][1])

        bombcell_labels = analyzer.get_sorting_property("bombcell")
        unit_locations = analyzer.get_extension("unit_locations").get_data()
        data = np.array([bombcell_labels, unit_locations[:, 0], unit_locations[:, 1]])

        location_info = pd.DataFrame(data.T, columns=["quality", "x_loc", "y_loc"])

        mua = location_info.query('quality == "mua"')
        good = location_info.query('quality == "good"')

        unit_ax.scatter(
            np.array(mua["x_loc"].values, dtype=float),
            np.array(mua["y_loc"].values, dtype=float),
            c="darkorange",
            alpha=0.45,
            s=25,
            edgecolors="none",
            label=f"MUA (n={len(mua)})",
            zorder=2,
        )

        unit_ax.scatter(
            np.array(good["x_loc"].values, dtype=float),
            np.array(good["y_loc"].values, dtype=float),
            c="#00C853",
            alpha=0.95,
            s=55,
            edgecolors="#004D20",
            linewidths=1.5,
            label=f"Good (n={len(good)})",
            zorder=3,
        )

        xmin, xmax = unit_ax.get_xlim()
        xmin -= 100
        xmax += 100
        unit_ax.set_xlim(xmin, xmax)

        unit_ax.set_aspect("equal")
        unit_ax.set_ylim(y_lims[0][0], y_lims[0][1])
        unit_ax.grid(axis="y")

        if shank_id == 0:
            day_title = f"Day {chunk_id}"
            unit_ax.set_title(day_title)
            amp_ax.set_title(day_title)
        else:
            amp_ax.set_title("")

        if chunk_id == 0:
            shank_text = f"Shank {shank_id}"
            unit_ax.set_ylabel(shank_text)
            amp_ax.set_ylabel(shank_text)

        if chunk_id > 0:
            unit_ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

            amp_ax.tick_params(
                axis="both",
                which="both",
                labelbottom=False,
                labelleft=False,
            )

            amp_ax.set_title("")
            amp_ax.set_xlabel("")
            amp_ax.set_ylabel("")

        for spine in unit_ax.spines.values():
            spine.set_visible(False)

amp_fig.tight_layout()
amp_fig.savefig("plots/spike_rasters.png")

unit_fig.tight_layout()
unit_fig.savefig("plots/unit_quality.png")
