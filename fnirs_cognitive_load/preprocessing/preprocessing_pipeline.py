
from abc import ABC, abstractmethod
from itertools import compress

import mne


class PreprocessingStep(ABC):

    @abstractmethod
    def transform(self, signal: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raise NotImplementedError("Subclasses should implement this method.")

    
class RawDetectorDistanceFilter(PreprocessingStep):
    
    def __init__(self, max_distance: float):
        self.max_distance = max_distance

    def transform(self, raw_intensity: mne.io.BaseRaw) -> mne.io.BaseRaw:
        picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
        dists = mne.preprocessing.nirs.source_detector_distances(
            raw_intensity.info, picks=picks
        )
        raw_intensity.pick(picks[dists > self.max_distance])
        return raw_intensity
    
class OpticalDensityConverter(PreprocessingStep):
    
    def transform(self, raw_intensity: mne.io.BaseRaw) -> mne.io.BaseRaw:
        return mne.preprocessing.nirs.optical_density(raw_intensity)

    
class ScalpCouplingIndexFilter(PreprocessingStep):
    
    def __init__(self, threshold: float):
        self.threshold = threshold

    def transform(self, raw_optical_density: mne.io.BaseRaw) -> mne.io.BaseRaw:
        sci = mne.preprocessing.nirs.scalp_coupling_index(raw_optical_density)
        raw_optical_density.info["bads"] = list(compress(raw_optical_density.ch_names, sci < self.threshold))
        return raw_optical_density
    
class BeerLambertLawConverter(PreprocessingStep):
    
    def transform(self, raw_optical_density: mne.io.BaseRaw) -> mne.io.BaseRaw:
        return mne.preprocessing.nirs.beer_lambert_law(raw_optical_density) 
    
class BandpassFilter(PreprocessingStep):
    
    def __init__(self, l_freq: float, h_freq: float, h_trans_bandwidth: float, l_trans_bandwidth: float):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        
    def transform(self, raw_hemoglobin: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw_hemoglobin.filter(
            self.l_freq, self.h_freq, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth
        )
        return raw_hemoglobin
    

class PreprocessingPipeline:
    def __init__(self, steps):
        self.steps = steps

    def transform(self, signal: mne.io.BaseRaw) -> mne.io.BaseRaw:
        for step in self.steps:
            signal = step.transform(signal)
        return signal