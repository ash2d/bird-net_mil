"""Tests for dataset parsing functions."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mil.datasets import (
    parse_strong_labels,
    parse_species_quality,
    extract_recording_id,
    _find_strong_label_file,
    events_to_labels,
    build_label_index,
    load_weak_labels_csv,
    extract_species_from_weak_csv,
    get_weak_labels_for_recording,
)


class TestParseSpeciesQuality:
    """Test species_quality parsing."""
    
    def test_parse_species_quality_high(self):
        """Test parsing species with high quality indicator."""
        species, quality = parse_species_quality("Boana_faber_H")
        assert species == "Boana_faber"
        assert quality == "H"
    
    def test_parse_species_quality_medium(self):
        """Test parsing species with medium quality indicator."""
        species, quality = parse_species_quality("Dendropsophus_minutus_M")
        assert species == "Dendropsophus_minutus"
        assert quality == "M"
    
    def test_parse_species_quality_low(self):
        """Test parsing species with low quality indicator."""
        species, quality = parse_species_quality("Scinax_fuscovarius_L")
        assert species == "Scinax_fuscovarius"
        assert quality == "L"
    
    def test_parse_species_quality_no_quality(self):
        """Test parsing species without quality indicator."""
        species, quality = parse_species_quality("Boana_faber")
        assert species == "Boana_faber"
        assert quality == ""
    
    def test_parse_species_quality_no_underscore(self):
        """Test parsing species name without underscores."""
        species, quality = parse_species_quality("Species")
        assert species == "Species"
        assert quality == ""
    
    def test_parse_species_quality_invalid_suffix(self):
        """Test parsing species with invalid quality suffix."""
        species, quality = parse_species_quality("Boana_faber_X")
        assert species == "Boana_faber_X"  # Entire string as species
        assert quality == ""
    
    def test_parse_species_quality_numeric_suffix(self):
        """Test parsing species with numeric suffix (not a quality)."""
        species, quality = parse_species_quality("Boana_faber_123")
        assert species == "Boana_faber_123"
        assert quality == ""
    
    def test_parse_species_quality_multiple_underscores(self):
        """Test parsing species with multiple underscores."""
        species, quality = parse_species_quality("Genus_species_subspecies_H")
        assert species == "Genus_species_subspecies"
        assert quality == "H"


class TestExtractRecordingId:
    """Test recording ID extraction from clip filenames."""
    
    def test_extract_new_format(self):
        """Test extraction from new format filename."""
        rec_id = extract_recording_id("INCT17_20200211_041500_0_3")
        assert rec_id == "INCT17_20200211_041500"
    
    def test_extract_new_format_longer(self):
        """Test extraction from new format with longer time values."""
        rec_id = extract_recording_id("INCT17_20200211_041500_57_60")
        assert rec_id == "INCT17_20200211_041500"
    
    def test_extract_old_format(self):
        """Test extraction from old format filename."""
        rec_id = extract_recording_id("INCT17_20200211_041500")
        assert rec_id == "INCT17_20200211_041500"
    
    def test_extract_complex_recording_id(self):
        """Test extraction when recording ID contains underscores."""
        rec_id = extract_recording_id("SITE_A_REC_001_0_3")
        assert rec_id == "SITE_A_REC_001"
    
    def test_extract_non_numeric_suffix(self):
        """Test extraction when suffix is not numeric."""
        rec_id = extract_recording_id("INCT17_20200211_041500_abc")
        assert rec_id == "INCT17_20200211_041500_abc"  # No change
    
    def test_extract_single_underscore(self):
        """Test extraction with single underscore."""
        rec_id = extract_recording_id("INCT17_001")
        assert rec_id == "INCT17_001"


class TestParseStrongLabels:
    """Test strong label file parsing."""
    
    def test_parse_strong_labels_new_format(self, tmp_path):
        """Test parsing strong labels in new format."""
        label_file = tmp_path / "INCT17_20200211_041500.txt"
        label_file.write_text(
            "0.5 2.3 Boana_faber_H\n"
            "1.2 4.5 Dendropsophus_minutus_M\n"
            "15.0 18.0 Scinax_fuscovarius_L\n"
        )
        
        events = parse_strong_labels(label_file)
        
        assert len(events) == 3
        assert events[0] == (0.5, 2.3, "Boana_faber", "H")
        assert events[1] == (1.2, 4.5, "Dendropsophus_minutus", "M")
        assert events[2] == (15.0, 18.0, "Scinax_fuscovarius", "L")
    
    def test_parse_strong_labels_no_quality(self, tmp_path):
        """Test parsing strong labels without quality indicators."""
        label_file = tmp_path / "INCT17_20200211_041500.txt"
        label_file.write_text("0.5 2.3 Boana_faber\n")
        
        events = parse_strong_labels(label_file)
        
        assert len(events) == 1
        assert events[0] == (0.5, 2.3, "Boana_faber", "")
    
    def test_parse_strong_labels_empty_file(self, tmp_path):
        """Test parsing empty strong labels file."""
        label_file = tmp_path / "REC_000001.txt"
        label_file.write_text("")
        
        events = parse_strong_labels(label_file)
        
        assert len(events) == 0
    
    def test_parse_strong_labels_nonexistent(self, tmp_path):
        """Test parsing nonexistent file returns empty list."""
        events = parse_strong_labels(tmp_path / "nonexistent.txt")
        assert events == []
    
    def test_parse_strong_labels_with_blank_lines(self, tmp_path):
        """Test parsing with blank lines."""
        label_file = tmp_path / "INCT17_20200211_041500.txt"
        label_file.write_text(
            "0.5 2.3 Boana_faber_H\n"
            "\n"
            "1.2 4.5 Dendropsophus_minutus_M\n"
        )
        
        events = parse_strong_labels(label_file)
        
        assert len(events) == 2


class TestFindStrongLabelFile:
    """Test strong label file finding."""
    
    def test_find_new_format_filename(self, tmp_path):
        """Test finding label file for new format embedding filename."""
        # Setup directory structure
        strong_root = tmp_path / "strong_labels"
        site_dir = strong_root / "SITE_A"
        site_dir.mkdir(parents=True)
        
        label_file = site_dir / "INCT17_20200211_041500.txt"
        label_file.write_text("0.5 2.3 Boana_faber_H\n")
        
        emb_dir = tmp_path / "embeddings" / "SITE_A"
        emb_dir.mkdir(parents=True)
        
        npz_path = emb_dir / "INCT17_20200211_041500_0_3.embeddings.npz"
        
        found = _find_strong_label_file(npz_path, strong_root)
        
        assert found == label_file
    
    def test_find_old_format_filename(self, tmp_path):
        """Test finding label file for old format embedding filename."""
        strong_root = tmp_path / "strong_labels"
        site_dir = strong_root / "SITE_A"
        site_dir.mkdir(parents=True)
        
        label_file = site_dir / "INCT17_20200211_041500.txt"
        label_file.write_text("0.5 2.3 Boana_faber_H\n")
        
        emb_dir = tmp_path / "embeddings" / "SITE_A"
        emb_dir.mkdir(parents=True)
        
        npz_path = emb_dir / "INCT17_20200211_041500.embeddings.npz"
        
        found = _find_strong_label_file(npz_path, strong_root)
        
        assert found == label_file
    
    def test_find_not_found(self, tmp_path):
        """Test returning None when label file not found."""
        strong_root = tmp_path / "strong_labels"
        strong_root.mkdir(parents=True)
        
        npz_path = tmp_path / "INCT17_20200211_041500_0_3.embeddings.npz"
        
        found = _find_strong_label_file(npz_path, strong_root)
        
        assert found is None


class TestEventsToLabels:
    """Test event to label conversion."""
    
    def test_events_to_labels_overlap(self):
        """Test label assignment with overlapping events."""
        events = [
            (0.5, 2.5, "Boana_faber", "H"),
            (3.0, 5.0, "Dendropsophus_minutus", "M"),
        ]
        label_index = {"Boana_faber": 0, "Dendropsophus_minutus": 1}
        
        # Clips: 0-3, 1-4, 2-5
        start_sec = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        end_sec = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        
        weak_labels, time_labels = events_to_labels(events, label_index, start_sec, end_sec)
        
        # Both species present in clip
        assert weak_labels[0] == 1.0  # Boana_faber
        assert weak_labels[1] == 1.0  # Dendropsophus_minutus
        
        # Boana_faber (0.5-2.5) overlaps with clips 0-3 and 1-4
        assert time_labels[0, 0] == 1.0  # Clip 0-3
        assert time_labels[1, 0] == 1.0  # Clip 1-4
        assert time_labels[2, 0] == 1.0  # Clip 2-5 (overlaps at 2.0-2.5)
        
        # Dendropsophus_minutus (3.0-5.0) overlaps with clips 1-4 and 2-5
        assert time_labels[0, 1] == 0.0  # Clip 0-3 (no overlap)
        assert time_labels[1, 1] == 1.0  # Clip 1-4
        assert time_labels[2, 1] == 1.0  # Clip 2-5
    
    def test_events_to_labels_unknown_species(self):
        """Test handling of unknown species."""
        events = [
            (0.5, 2.5, "Unknown_species", "H"),
        ]
        label_index = {"Boana_faber": 0}
        
        start_sec = np.array([0.0], dtype=np.float32)
        end_sec = np.array([3.0], dtype=np.float32)
        
        weak_labels, time_labels = events_to_labels(events, label_index, start_sec, end_sec)
        
        # Unknown species should be ignored
        assert weak_labels[0] == 0.0
        assert time_labels[0, 0] == 0.0


class TestBuildLabelIndex:
    """Test label index building."""
    
    def test_build_from_strong_root(self, tmp_path):
        """Test building label index from strong label files."""
        strong_root = tmp_path / "strong_labels"
        site_dir = strong_root / "SITE_A"
        site_dir.mkdir(parents=True)
        
        label_file = site_dir / "REC_000001.txt"
        label_file.write_text(
            "0.5 2.3 Boana_faber_H\n"
            "1.0 3.0 Dendropsophus_minutus_M\n"
        )
        
        label_index = build_label_index(strong_root=strong_root)
        
        assert len(label_index) == 2
        assert "Boana_faber" in label_index
        assert "Dendropsophus_minutus" in label_index
    
    def test_build_from_species_list(self):
        """Test building label index from explicit species list."""
        species = ["Species_A", "Species_B", "Species_C"]
        label_index = build_label_index(species_list=species)
        
        assert len(label_index) == 3
        # Should be alphabetically sorted
        assert label_index["Species_A"] == 0
        assert label_index["Species_B"] == 1
        assert label_index["Species_C"] == 2


class TestWeakLabels:
    """Test weak label CSV loading and processing."""
    
    def test_load_weak_labels_csv(self, tmp_path):
        """Test loading weak labels from CSV."""
        csv_file = tmp_path / "weak_labels.csv"
        csv_content = """MONITORING_SITE,AUDIO_FILE_ID,SPECIES_Boana_faber,SPECIES_Dendropsophus_minutus
SITE_A,REC_001,1,0
SITE_A,REC_002,0,1
SITE_B,REC_003,1,1
"""
        csv_file.write_text(csv_content)
        
        df = load_weak_labels_csv(csv_file)
        
        assert len(df) == 3
        assert "MONITORING_SITE" in df.columns
        assert "AUDIO_FILE_ID" in df.columns
        assert "SPECIES_Boana_faber" in df.columns
        assert "SPECIES_Dendropsophus_minutus" in df.columns
    
    def test_load_weak_labels_csv_missing_columns(self, tmp_path):
        """Test loading CSV with missing required columns."""
        csv_file = tmp_path / "weak_labels.csv"
        csv_content = """SITE,FILE,SPECIES_Boana_faber
SITE_A,REC_001,1
"""
        csv_file.write_text(csv_content)
        
        with pytest.raises(ValueError):
            load_weak_labels_csv(csv_file)
    
    def test_extract_species_from_weak_csv(self, tmp_path):
        """Test extracting species names from CSV columns."""
        csv_file = tmp_path / "weak_labels.csv"
        csv_content = """MONITORING_SITE,AUDIO_FILE_ID,SPECIES_Boana_faber,SPECIES_Dendropsophus_minutus,SPECIES_Scinax_fuscovarius
SITE_A,REC_001,1,0,1
"""
        csv_file.write_text(csv_content)
        
        df = load_weak_labels_csv(csv_file)
        species = extract_species_from_weak_csv(df)
        
        assert len(species) == 3
        assert "Boana_faber" in species
        assert "Dendropsophus_minutus" in species
        assert "Scinax_fuscovarius" in species
        # Should be sorted
        assert species == sorted(species)
    
    def test_get_weak_labels_for_recording_found(self, tmp_path):
        """Test getting weak labels for a recording that exists."""
        csv_file = tmp_path / "weak_labels.csv"
        csv_content = """MONITORING_SITE,AUDIO_FILE_ID,SPECIES_Boana_faber,SPECIES_Dendropsophus_minutus
SITE_A,REC_001,1,0
SITE_A,REC_002,0,1
"""
        csv_file.write_text(csv_content)
        
        df = load_weak_labels_csv(csv_file)
        label_index = {"Boana_faber": 0, "Dendropsophus_minutus": 1}
        
        labels = get_weak_labels_for_recording(df, "REC_001", label_index)
        
        assert labels[0] == 1.0  # Boana_faber
        assert labels[1] == 0.0  # Dendropsophus_minutus
    
    def test_get_weak_labels_for_recording_not_found(self, tmp_path):
        """Test getting weak labels for a recording that doesn't exist."""
        csv_file = tmp_path / "weak_labels.csv"
        csv_content = """MONITORING_SITE,AUDIO_FILE_ID,SPECIES_Boana_faber
SITE_A,REC_001,1
"""
        csv_file.write_text(csv_content)
        
        df = load_weak_labels_csv(csv_file)
        label_index = {"Boana_faber": 0}
        
        labels = get_weak_labels_for_recording(df, "REC_999", label_index)
        
        # Should return all zeros
        assert labels[0] == 0.0
    
    def test_build_label_index_from_weak_csv(self, tmp_path):
        """Test building label index from weak CSV."""
        csv_file = tmp_path / "weak_labels.csv"
        csv_content = """MONITORING_SITE,AUDIO_FILE_ID,SPECIES_Boana_faber,SPECIES_Dendropsophus_minutus,SPECIES_Scinax_fuscovarius
SITE_A,REC_001,1,0,1
"""
        csv_file.write_text(csv_content)
        
        label_index = build_label_index(weak_csv=csv_file)
        
        assert len(label_index) == 3
        assert "Boana_faber" in label_index
        assert "Dendropsophus_minutus" in label_index
        assert "Scinax_fuscovarius" in label_index
        # Should be sorted
        assert label_index["Boana_faber"] == 0
        assert label_index["Dendropsophus_minutus"] == 1
        assert label_index["Scinax_fuscovarius"] == 2



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
