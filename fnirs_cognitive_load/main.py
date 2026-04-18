import matplotlib.pyplot as plt
from mne.preprocessing.nirs import beer_lambert_law, optical_density

from fnirs_cognitive_load.config import DATA_RAW
from fnirs_cognitive_load.preprocessing.snirf_database_helper import SnirfDatabaseHelper


def main():
    snirf_db = SnirfDatabaseHelper(DATA_RAW / "Walking")
    
    snirf_file = snirf_db.get_snirf_files()[0]
    
    raw_intensity = snirf_db.read_snirf_file(snirf_file)
    
    
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(211)
    raw_intensity.plot(scalings='auto', show=False, axes=ax1)
    ax1.set_title('Raw Intensity')
    fig.show()