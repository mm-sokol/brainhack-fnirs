from logging import getLogger
from pathlib import Path

import mne
import pandas as pd

logger = getLogger(__name__)

class SnirfDatabaseHelper():
    def __init__(self, root, event_map_dir="PsychoPy"):

        self.trial_ann_list = ['mental']
        self.rest_ann_list = ['rest']
        self.full_ann_list = self.trial_ann_list + self.rest_ann_list
        self.root = Path(root)
        self.event_map_root = self.root / event_map_dir

    def get_sub_name_from_path(self, path):
        name_parts = path.name.split('_')
        logger.info(name_parts)
        sub = name_parts[3] + '_' + name_parts[4] + '_' + name_parts[5] 
        logger.info(f"Extracted subject name: {sub} from path: {path}")
        return sub

    def get_snirf_files(self):
        files = [f for f in self.root.glob("*.snirf") if f.is_file()]
        logger.info(f"Found {len(files)} .snirf files in {self.root}")
        return files
    
    def get_event_map(self, sub_name):
        event_map_path = self.event_map_root.glob(f"{sub_name}*.csv")
        
        csv_files = list(self.event_map_root.glob(f"{sub_name}*.csv"))

        if event_map_path is None or len(csv_files) == 0:
            logger.warning(f"No event map file found for subject {sub_name} in {self.event_map_root}.")
            return None
        
        event_map_path = csv_files[0]
        logger.info(f"Found event map file: {event_map_path} for subject {sub_name}")
        
        event_map_df = pd.read_csv(event_map_path)

        return event_map_df
        
    def read_snirf_file(self, snirf_path):
        raw_intensity = mne.io.read_raw_snirf(snirf_path, preload=True, verbose=False)
        
        return raw_intensity