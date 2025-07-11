# Korean Phoneme Onset Detection for EEG-Linguistic Feature Analysis

A Python tool for extracting phoneme onset timing from Korean speech audio, designed for EEG research investigating linguistic feature processing.

## Overview

While traditional EEG-speech studies focus on acoustic features like speech envelope, this project extracts higher-level **linguistic features** - specifically phoneme onsets - to investigate cognitive language processing capabilities. The tool provides precise timing information of phoneme boundaries, enabling researchers to correlate neural responses with linguistic events.

## Features

- **Automatic Speech Recognition (ASR)** using Korean pre-trained models
- **Phoneme onset detection** with millisecond precision 
- **Multiple output formats** for different analysis needs:
  - CSV files with phoneme timing data
  - Spike trains at 50Hz and 64Hz sampling rates (for EEG synchronization)
  - Individual phoneme audio segments
  - Mel-spectrogram visualizations with onset markers
- **Configurable processing options** via YAML configuration
- **Automatic model download** on first run

## Installation

1. Download this repository:

Click the green Code button on this repository page
Select Download ZIP
Extract the downloaded file to your desired location

2. Install required packages:
```bash
pip install torch transformers librosa pandas numpy matplotlib soundfile scipy jamo pyyaml
```

3. Place your audio files (`.wav` format) in the `stimuli/` directory

## Usage

1. **Configure settings** in `config.yaml`:
```yaml
# Input stimuli files
stim_dir: './stimuli'           # Stimulus directory
keyword: 'tstory'               # Files only including keyword (e.g. 'tstory' : only files with tstory is processed)
result_dir: './results'         # Results directory

# Output configuration
result_dir: './results'         # Results directory

...

# Processing options
enable_visualization: true      # Generate mel-spectrogram with phoneme markers
enable_segmentation: true       # Save individual phoneme audio segments
enable_feature_extraction: true # Convert to spike trains and save as .npy

# Feature extraction settings
spike_train:
  source_rate: 50              # Original phoneme detection rate (Hz)
  target_rate: 64              # Target EEG sampling rate (Hz)
  duration: 120                # Duration in seconds
  resampling_method: 'nearest' # 'nearest' or 'window'
```

2. **Run the detection**:
```bash
python phoneme_detection.py
```

**Note**: On first run, the tool will automatically download pre-trained Korean models (~1-2GB), which may take several minutes depending on your internet connection.

## Output Structure

The tool generates organized outputs for each audio file:

```
results/
└── audio_filename/
    ├── ASR_result_audio_filename.txt                                # Speech recognition text
    ├── character_onset_times_audio_filename.csv                     # Phoneme timing data
    ├── plots/
    │   └── mel_spectrogram_phoneme_onsets_audio_filename.pdf        # Visualization
    ├── segments/
    │   └── segment_idx_001_onset_0.20_char_ㄱ.wav                   # Individual phonemes
    └── features/
        ├── audio_filename_phoneme_onset_50hz.npy                   # 50Hz spike train
        └── audio_filename_phoneme_onset_64hz.npy                   # 64Hz spike train
```

### Output File Descriptions

- **CSV file**: Contains phoneme characters and their onset times in seconds
- **TXT file**: Plain text ASR transcription result
- **PDF plots**: Mel-spectrograms with vertical lines marking phoneme onsets
- **WAV segments**: Individual audio clips for each detected phoneme
- **NPY features**: Binary spike trains synchronized to EEG sampling rates

## Models Used

This tool uses pre-trained Korean models from Hugging Face:
- **Speech Recognition**: `42MARU/ko-spelling-wav2vec2-conformer-del-1s`
- **Language Model**: `42MARU/ko-ctc-kenlm-spelling-only-wiki`

Models are automatically downloaded and cached locally on first use.

## Configuration Options

The `config.yaml` file allows you to customize:

- **Input/Output paths**: Specify directories for audio files and results
- **File filtering**: Process only files containing specific keywords
- **Processing modules**: Enable/disable visualization, segmentation, and feature extraction
- **EEG synchronization**: Adjust sampling rates for neural data correlation
- **ASR parameters**: Modify beam search width and processing device

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Librosa
- Other dependencies listed in installation section

## Use Cases

This tool is particularly useful for:
- EEG studies investigating phonological processing
- Speech-brain coupling analysis at the phoneme level
- Temporal alignment of neural responses with linguistic events
