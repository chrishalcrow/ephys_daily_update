from pathlib import Path
import json

import numpy as np
import probeinterface as pi
import spikeinterface.full as si
from aeon.dj_pipeline.utils.spike_sorting_utils import read_bonsai_onix_arrow

si.set_global_job_kwargs(n_jobs=1)

with open('config.json') as f:
    config = json.load(f)

experiment_path = config['experiment_path']
chunks = config['chunks']
probe_path = config['probe_path']

chunk_filepaths = [Path(experiment_path) / chunk for chunk in chunks]

processed_folder = Path("processed")

probe = pi.read_probeinterface(probe_path)

for chunk_filepath in chunk_filepaths:
    
    rec = si.unsigned_to_signed(
        read_bonsai_onix_arrow(
            str(chunk_filepath),
            sampling_frequency=30_000,
            gain_to_uV=3.05176,
            offset_to_uV=0,
        )
    )
    
    rec.set_probegroup(probe)
    
    for shank_id in [0, 1, 2, 3]:
        grouped_rec = rec.split_by("group")[shank_id]
    
        sorter_output_folder = processed_folder / (
            chunk_filepath.name.split(".")[0] + f"_lupin_sorter_output_s{shank_id}"
        )
        analyzer_output_folder = processed_folder / (
            chunk_filepath.name.split(".")[0] + f"_lupin_analyzer_s{shank_id}"
        )

        if Path(str(analyzer_output_folder) + ".zarr").is_dir():
            print("skipping ...")
            continue
        
        sort = si.run_sorter(
            "lupin",
            grouped_rec,
            folder=sorter_output_folder,
            apply_motion_correction=False,
            verbose=True,
            remove_existing_folder=True,
            detect_threshold=10.0,
            with_output=True,
        )
        # sort = si.load('/Users/christopherhalcrow/Work/fromgit/ephys_daily_update/processed/NeuropixelsV2_ProbeB_2_lupin_sorter_output/')
    
        # analyzer = si.load(analyzer_output_folder)
    
        analyzer = si.create_sorting_analyzer(
            sort,
            si.bandpass_filter(si.common_reference(grouped_rec)),
            folder=analyzer_output_folder,
            overwrite=True,
            format="zarr",
        )
    
        analyzer.compute(
            [
                "random_spikes",
                "noise_levels",
                "templates",
                "unit_locations",
                "spike_locations",
                "spike_amplitudes",
                "quality_metrics",
                "template_metrics",
                "correlograms",
            ]
        )
    
        unit_quality = si.bombcell_label_units(analyzer)
        analyzer.set_sorting_property(
            key="bombcell", values=np.array(unit_quality["bombcell_label"], dtype=str)
        )
