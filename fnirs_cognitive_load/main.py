from itertools import compress

import matplotlib.pyplot as plt
import mne
from mne.preprocessing.nirs import beer_lambert_law, optical_density, scalp_coupling_index

from fnirs_cognitive_load.config import DATA_RAW, FIG
from fnirs_cognitive_load.preprocessing.snirf_database_helper import SnirfDatabaseHelper


def main():
    
    snirf_data = SnirfDatabaseHelper(DATA_RAW / "Walking")
    plt.switch_backend('QtAgg') 

    for snirf_file in snirf_data.get_snirf_files()[10:]:
        subject_name = snirf_data.get_sub_name_from_path(snirf_file)
        
        raw_intensity = snirf_data.read_snirf_file(snirf_file)
        picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
        dists = mne.preprocessing.nirs.source_detector_distances(
            raw_intensity.info, picks=picks
        )
        raw_intensity.pick(picks[dists > 0.01])
        
        raw_optical_density = optical_density(raw_intensity)
        
        sci = scalp_coupling_index(raw_optical_density)
        
        fig, ax = plt.subplots(layout="constrained")
        # ax.hist(sci)
        # ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])   
        # fig.savefig(FIG / f"sci_histogram_{subject_name}.png")
                
        
        raw_optical_density.info["bads"] = list(compress(raw_optical_density.ch_names, sci < 0.7))
        raw_hemoglobin = beer_lambert_law(raw_optical_density)
        # raw_hemoglobin.plot(block=True, show=True, title=f"Hemoglobin Concentration for Subject {subject_name}", duration=19*40, bad_color='gray')
        # plt.show(block=True)
        
        raw_haemo_unfiltered = raw_hemoglobin.copy()
        raw_hemoglobin.filter(0.01, 0.3, h_trans_bandwidth=0.2, l_trans_bandwidth=0.005)
        for when, _raw in dict(Before=raw_haemo_unfiltered, After=raw_hemoglobin).items():
            fig = _raw.compute_psd(fmin=0, fmax=3).plot(
                average=True, amplitude=False, picks="data", exclude="bads"
            )
            fig.suptitle(f"{when} filtering", weight="bold", size="x-large")
               
            fig.savefig(FIG / f"psd_comparison_{when.lower()}_{subject_name}.png") 
    
        events, event_dict = mne.events_from_annotations(raw_hemoglobin)
        fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_hemoglobin.info["sfreq"])
        fig.savefig(FIG / f"event_timeline_{subject_name}.png")


        reject_criteria = dict(hbo=20e-6, hbr=20e-6)
        tmin, tmax = -10, 16

        epochs = mne.Epochs(
            raw_hemoglobin,
            events,
            event_id=event_dict,
            tmin=tmin,
            tmax=tmax,
            reject=reject_criteria,
            reject_by_annotation=True,
            proj=True,
            baseline=(None, 0),
            preload=True,
            detrend=None,
            verbose=True,
        )
       
        fig = epochs.plot_drop_log()
        fig.savefig(FIG / f"epoch_drop_log_{subject_name}.png")
    
        task = "2"
        fig = epochs[task].plot_image(
            combine="mean",
            vmin=-1,
            vmax=1,
            ts_args=dict(ylim=dict(hbo=[-0.2, 0.2], hbr=[-0.2, 0.2])),
        )
        
        
        break
    
    
if '__main__' == __name__:
    main()
        