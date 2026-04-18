from logging import getLogger
from pathlib import Path

import mne
import panads as pd

from fnirs_cognitive_load.preprocessing.preprocessing_pipeline import PreprocessingPipeline

logger = getLogger(__name__)

class EventDataset:
    def __init__(self, 
                 root: Path, 
                 subdirs: list[str]=None, 
                 preprocessing: PreprocessingPipeline=None, 
                 event_window: tuple[float, float]=(-10, 16), 
                 reject_criteria: dict[str, float]=dict(hbo=80e-6, hbr=80e-6)
                ):

        self.data_paths = [root / subdir for subdir in subdirs] if subdirs else [root]
        self.preprocessing = preprocessing
        self.subjects = self._init_subject_list()
        self.subject_files = self._map_subjects_to_files()
        self.event_window = event_window
        self.reject_criteria = reject_criteria
        
        self.current_subject_idx = 0
        self.current_epochs = None


    def get_subjects(self) -> list[str]:
        return self.subjects
    
    def get_subject_data(self, subject: str) -> dict[str, Path]:
        
        if subject not in self.subject_files:
            logger.error(f"Subject {subject} not found in dataset.")
            return {}
        
        subject_data = {}
        for subset, file_paths in self.subject_files[subject].items():
            if subset not in self.data_paths:
                logger.error(f"Data path {subset} for subject {subject} is not in the initialized data paths.")
                return {}
            
            sfnirf_file = file_paths.get("snirf")
            csv_file = file_paths.get("csv")
            
            if not sfnirf_file:
                logger.error(f"No .snirf file found for subject {subject} in {subset}.")
                return {}
            if not csv_file:
                logger.error(f"No .csv file found for subject {subject} in {subset}.")
                return {}
            
            subject_data[subset] = {"snirf": self._get_raw_data(sfnirf_file), "csv": self._get_event_data(csv_file)}
            
        return subject_data
            


    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_subject_idx >= len(self.subjects):
            raise StopIteration
        
        subject = self.subjects[self.current_subject_idx]
        subject_data = self.get_subject_data(subject)
        if not subject_data:
            logger.error(f"Skipping subject {subject} due to missing data.")
            self.current_subject_idx += 1
            return self.__next__()
        
        subject_epochs = self.get_subject_epochs(subject_data)
        self.current_subject_idx += 1
        
        return subject, subject_epochs
        
    
    def get_subject_epochs(self, subject_data: dict[str, dict[str, Path]]) -> mne.Epochs:
        
        subject_epochs = {}
        for subset, data in subject_data.items():
            raw_data = data.get("snirf")
            event_data = data.get("csv")
            if raw_data is None or event_data is None:
                logger.error(f"Skipping subject in subset {subset} due to missing raw or event data.")
                continue
            if self.preprocessing:
                raw_data = self.preprocessing.transform(raw_data)
                
            events, event_dict = mne.events_from_annotations(raw_data)
            tiral_start_times = event_data.at[0, 'trial.started']
            raw_data.crop(tmin=tiral_start_times, tmax=None)
            epochs = mne.Epochs(
                raw_data,
                events,
                event_id=event_dict,
                tmin=self.event_window[0],
                tmax=self.event_window[1],
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=(None, 0),
                preload=True,
                detrend=None,
                verbose=True,
            )
            subject_epochs[subset] = epochs
        return subject_epochs


            
            
            
    # --------- Private methods ---------
    def _get_subject_name(self, subject_path: Path) -> str:
        pathname = subject_path.name
        parts = pathname.split('_')
        if len(parts) < 6:
            raise ValueError(f"Unexpected subject path format: {pathname}")
        return f"{parts[3]}_{parts[4]}_{parts[5]}"
    
    def _init_subject_list(self) -> list[str]:
        subject_list = set()
        for data_path in self.data_paths:
            snirsf_files = list(data_path.glob("*.snirf"))
            subject_list.update(self._get_subject_name(snirf) for snirf in snirsf_files)
        return list(subject_list)
    
    def _map_subjects_to_files(self) -> dict[str, list[Path]]:
        subject_files = {subject: {} for subject in self.subjects}
        
        for subject in self.subjects:
            for data_path in self.data_paths:
                subject_files[subject][data_path.name] = {}
                snirsf_files = list(data_path.glob(f"*{subject}*.snirf"))
                if snirsf_files and len(snirsf_files) > 0:
                    subject_files[subject][data_path.name]["snirf"] = snirsf_files[0]
                else:
                    logger.error(f"No .snirf file found for subject {subject} in {data_path}")
                    
                csv_files = list(data_path.glob(f"{subject}*.csv"))
                if csv_files and len(csv_files) > 0:
                    subject_files[subject][data_path.name]["csv"] = csv_files[0]
                else:
                    logger.error(f"No .csv file found for subject {subject} in {data_path}")
            
        return subject_files
    
    
    def _get_raw_data(self, snirf_file: Path) -> mne.io.BaseRaw:
        return mne.io.read_raw_snirf(snirf_file, preload=True)
    
    def _get_event_data(self, csv_file: Path) -> dict:
        event_map_df = pd.read_csv(csv_file)
        return event_map_df[3:22]
        