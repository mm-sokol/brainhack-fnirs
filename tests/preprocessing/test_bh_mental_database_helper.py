import pytest

from fnirs_cognitive_load.config import DATA_RAW
from fnirs_cognitive_load.preprocessing.snirf_database_helper import SnirfDatabaseHelper


@pytest.fixture
def sample_database():
    return SnirfDatabaseHelper(DATA_RAW / "Walking")

def test_get_snirf_files(sample_database):
    snirf_files = sample_database.get_snirf_files()
    assert len(snirf_files) > 0, "No .snirf files found in the specified directory. Please check the path and file permissions."
    
    subject = sample_database.get_sub_name_from_path(snirf_files[0])
    assert subject is not None, "Failed to extract subject name from .snirf file path. Please check the filename format."
    assert len(subject.split('_')) == 3, "Extracted subject name does not have the expected format (e.g., '3_PA_WALKING')."
    assert "WALKING" in subject, "Extracted subject name does not contain the expected task identifier 'WALKING'."
    
    
def test_get_event_map(sample_database):
    snirf_files = sample_database.get_snirf_files()
    assert len(snirf_files) > 0, "No .snirf files found in the specified directory. Please check the path and file permissions."
    
    subject = sample_database.get_sub_name_from_path(snirf_files[0])
    event_map_df = sample_database.get_event_map(subject)
    
    assert event_map_df is not None, f"No event map found for subject {subject}. Please check if the corresponding CSV file exists in the event map directory."
    
def test_read_snirf_file(sample_database):
    
    snirf_file = sample_database.get_snirf_files()[0]
    raw_intensity = sample_database.read_snirf_file(snirf_file)
    assert raw_intensity is not None, "Failed to read .snirf file. Please check the file format and contents."
    assert raw_intensity.info['nchan'] > 0, "The .snirf file does not contain any channels. Please check the file contents."
    assert raw_intensity.info['sfreq'] > 0, "The .snirf file does not contain a valid sampling frequency. Please check the file contents."
    