import os
import re
import unicodedata
import warnings
from glob import glob
from time import time

import jamo
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from scipy.interpolate import interp1d
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    Wav2Vec2ProcessorWithLM,
)
from transformers.pipelines import AutomaticSpeechRecognitionPipeline


class PhonemeDetection:
    def __init__(self, config):
        # Basic configuration
        self.stim_dir = config["stim_dir"]
        self.keyword = config["keyword"]
        self.result_dir = config["result_dir"]

        # Audio processing settings
        self.sampling_rate = config["sampling_rate"]
        self.beam_width = config["asr"]["beam_width"]
        self.device = config["asr"]["device"]

        # Processing options
        self.enable_visualization = config["enable_visualization"]
        self.enable_segmentation = config["enable_segmentation"]
        self.enable_feature_extraction = config["enable_feature_extraction"]

        # Feature extraction settings
        spike_config = config["spike_train"]
        self.source_rate = spike_config["source_rate"]
        self.target_rate = spike_config["target_rate"]
        self.duration = spike_config["duration"]
        self.resampling_method = spike_config["resampling_method"]

        # Load models
        self._load_models()

        # Get stimulus files
        self.stim_fpath_list = sorted(glob(os.path.join(self.stim_dir, "*.wav")))
        self.stim_fpath_list = [stim_fpath for stim_fpath in self.stim_fpath_list if self.keyword in stim_fpath]
        self.stim_fname_list = [os.path.basename(stim_fpath) for stim_fpath in self.stim_fpath_list]
        print(f'Stimuli files: {self.stim_fname_list}')

    def _load_models(self):
        """Load models, download if not exists"""
        model_path = "./saved_model"
        main_model_name = "42MARU/ko-spelling-wav2vec2-conformer-del-1s"
        lm_model_name = "42MARU/ko-ctc-kenlm-spelling-only-wiki"

        if not os.path.exists(model_path):
            print("Model not found. Downloading from Hugging Face...")

            # Download main model components
            self.model = AutoModelForCTC.from_pretrained(main_model_name)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(main_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(main_model_name)

            # Download language model processor
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(lm_model_name)

            # Save locally
            print("Saving model locally...")
            self.model.save_pretrained(model_path)
            self.feature_extractor.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            self.processor.save_pretrained(model_path)
        else:
            print("Loading local model...")
            self.model = AutoModelForCTC.from_pretrained(model_path)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_path)

        # Create ASR pipeline
        self.asr_pipeline = AutomaticSpeechRecognitionPipeline(
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            decoder=self.processor.decoder,
            device=self.device
        )

    def create_output_directories(self, result_path):
        """Create output directories for different types of results"""
        directories = [result_path]

        if self.enable_visualization:
            directories.append(os.path.join(result_path, "plots"))
        if self.enable_segmentation:
            directories.append(os.path.join(result_path, "segments"))
        if self.enable_feature_extraction:
            directories.append(os.path.join(result_path, "features"))

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _read_audio(self, audio_fpath):
        """Read audio file and return raw data"""
        print('Reading audio...')
        raw_data, _ = librosa.load(audio_fpath, sr=self.sampling_rate)
        print('Reading audio done!')
        return raw_data

    def asr(self, audio_fpath, audio_fname_wo_wav, result_path):
        """Perform Automatic Speech Recognition"""
        print('----------------- ASR Procedure -----------------')
        raw_data = self._read_audio(audio_fpath)
        kwargs = {"decoder_kwargs": {"beam_width": self.beam_width}}
        print('Making ASR pipeline...')
        pred = self.asr_pipeline(inputs=raw_data, **kwargs)["text"]
        print('Making ASR pipeline done!')
        result = unicodedata.normalize("NFC", pred)
        print('ASR Result:\n', result)

        # Save ASR result
        with open(f"{result_path}/ASR_result_{audio_fname_wo_wav}.txt", "w", encoding="utf-8") as f:
            f.write(result)
        return result

    def _double_filtering(self, pred_char_list):
        """Filter consecutive identical phonemes"""
        count = 0
        choseong_stack = 0
        jungseong_stack = 0
        jongseong_stack = 0
        df_list = []

        for char_idx, char in enumerate(pred_char_list):
            if char != '<pad>' and char != '|':
                # Check Korean phoneme types
                if bool(re.match(r'[\u1100-\u1112]', char)):  # Initial consonants
                    choseong_stack += 1
                else:
                    choseong_stack = 0
                if bool(re.match(r'[\u1160-\u11A7]', char)):  # Vowels
                    jungseong_stack += 1
                else:
                    jungseong_stack = 0
                if bool(re.match(r'[\u11A8-\u11FF]', char)):  # Final consonants
                    jongseong_stack += 1
                else:
                    jongseong_stack = 0

                # Only keep if not too many consecutive phonemes
                if choseong_stack < 2 and jungseong_stack < 2 and jongseong_stack < 2:
                    count += 1
                    onset_time = char_idx * 0.02
                    print(f'Character: {char}, Onset time (sec): {onset_time}')
                    df_list.append({'Character': char, 'Onset': onset_time})

        return df_list, count

    def _split_hangul_to_jamo(self, text):
        """Split Hangul characters into individual Jamo components"""
        result = []
        for char in text:
            if '가' <= char <= '힣':
                jamo_chars = list(jamo.hangul_to_jamo(char))
                result.extend(jamo_chars)
            else:
                result.append(char)
        return result

    def phoneme_detect(self, audio_fpath, audio_fname_wo_wav, result_path):
        """Detect phoneme onset times"""
        print('----------------- Phoneme Detection Procedure -----------------')
        raw_data = self._read_audio(audio_fpath)

        print('Making Tokens...')
        char_list = self.processor.tokenizer.get_vocab()
        char_list = sorted(char_list, key=lambda x: char_list[x])
        print('Making Tokens done!')

        print('Processing audio and predicting phonemes onset...')
        input_values = self.processor(raw_data, return_tensors="pt", sampling_rate=self.sampling_rate).input_values
        with torch.no_grad():
            logits = self.model(input_values).logits.cpu().numpy()[0]

        pred_char_list = [char_list[mx_idx] for mx_idx in np.argmax(logits, axis=1)]
        print('Processing audio and predicting phonemes done!')

        # Filter and save results
        df_list, count = self._double_filtering(pred_char_list)
        df = pd.DataFrame(df_list)
        df.to_csv(f'{result_path}/character_onset_times_{audio_fname_wo_wav}.csv',
                  index=False, encoding='utf-8-sig')

        return df

    def visualize_phoneme_onsets(self, audio_fpath, df, result_path, audio_fname_wo_wav, end_time=5):
        """Generate mel-spectrogram with phoneme onset markers"""
        if not self.enable_visualization:
            return

        print('Generating visualization...')

        # Load and trim audio
        y, sr = librosa.load(audio_fpath, sr=self.sampling_rate)
        y = y[:end_time * sr]

        # Extract phoneme onsets within time range
        phoneme_onsets = df[df['Onset'] < end_time]['Onset'].tolist()
        phoneme_chars = df[df['Onset'] < end_time]['Character'].tolist()

        # Calculate mel-spectrogram
        n_fft = 2048
        hop_length = 320
        n_mels = 128
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Create plot
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                 x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel-spectrogram with Phoneme Onset Markers")

        # Add onset markers
        for onset in phoneme_onsets:
            plt.axvline(x=onset, color="g", linestyle="--", linewidth=1.5, alpha=0.8)

        plt.xlabel("Time (s)")
        plt.ylabel("Mel Frequency")

        # Save plot
        plot_path = os.path.join(result_path, "plots")
        plt.savefig(f"{plot_path}/mel_spectrogram_phoneme_onsets_{audio_fname_wo_wav}.pdf",
                    format='pdf', bbox_inches='tight')
        plt.close()
        print('Visualization saved!')

    def segment_audio_by_phonemes(self, audio_fpath, df, result_path, audio_fname_wo_wav):
        """Segment audio by phoneme onsets and save individual files"""
        if not self.enable_segmentation:
            return

        print('Segmenting audio by phonemes...')

        # Load audio
        y, sr = librosa.load(audio_fpath, sr=self.sampling_rate)

        # Get onset times and characters
        phoneme_onsets = df['Onset'].tolist()
        phoneme_chars = df['Character'].tolist()

        # Create segments directory
        segments_path = os.path.join(result_path, "segments")

        # Segment and save each phoneme
        for idx, (onset, char) in enumerate(zip(phoneme_onsets, phoneme_chars)):
            start_sample = int(onset * sr)

            if idx == len(phoneme_onsets) - 1:
                # Last segment goes to end of audio
                y_segment = y[start_sample:]
            else:
                # Segment until next phoneme onset
                end_sample = int(phoneme_onsets[idx + 1] * sr)
                y_segment = y[start_sample:end_sample]

            # Save segment
            segment_fname = f"segment_idx_{idx:03d}_onset_{onset:.2f}_char_{char}.wav"
            segment_fpath = os.path.join(segments_path, segment_fname)
            sf.write(segment_fpath, y_segment, samplerate=sr)

        print(f'Audio segmentation completed! {len(phoneme_onsets)} segments saved.')

    def convert_to_spike_train(self, df):
        """Convert phoneme onsets to spike train format"""
        # Create 50Hz spike train
        phoneme_onset_50hz = np.zeros(self.source_rate * self.duration)

        for onset_val in df['Onset']:
            # Add small epsilon to handle floating point errors
            idx_check = int((onset_val + 1e-7) // (1 / self.source_rate))
            if idx_check < self.source_rate * self.duration:
                phoneme_onset_50hz[idx_check] = 1

        # Resample to target rate (64Hz)
        phoneme_onset_64hz = self._resample_spike_train(phoneme_onset_50hz)

        return phoneme_onset_50hz, phoneme_onset_64hz

    def _resample_spike_train(self, spike_train_50hz):
        """Resample spike train from 50Hz to target rate"""
        duration = len(spike_train_50hz) / self.source_rate
        time_50hz = np.linspace(0, duration, len(spike_train_50hz), endpoint=False)
        time_target = np.linspace(0, duration, int(duration * self.target_rate), endpoint=False)

        if self.resampling_method == 'nearest':
            interp_func = interp1d(time_50hz, spike_train_50hz, kind='nearest',
                                   bounds_error=False, fill_value=0)
            spike_train_target = interp_func(time_target)
        elif self.resampling_method == 'window':
            spike_train_target = np.zeros(len(time_target))
            for i, t in enumerate(time_target):
                window_start = t - (1 / (2 * self.target_rate))
                window_end = t + (1 / (2 * self.target_rate))
                spikes_in_window = spike_train_50hz[(time_50hz >= window_start) & (time_50hz < window_end)]
                spike_train_target[i] = 1 if np.any(spikes_in_window) else 0
        else:
            raise ValueError("Method should be 'nearest' or 'window'")

        return spike_train_target.astype(int)

    def save_features(self, spike_train_50hz, spike_train_64hz, result_path, audio_fname_wo_wav):
        """Save spike train features as numpy arrays"""
        if not self.enable_feature_extraction:
            return

        print('Saving spike train features...')

        features_path = os.path.join(result_path, "features")

        # Save both 50Hz and 64Hz versions
        np.save(f'{features_path}/{audio_fname_wo_wav}_phoneme_onset_50hz.npy', spike_train_50hz)
        np.save(f'{features_path}/{audio_fname_wo_wav}_phoneme_onset_64hz.npy', spike_train_64hz)

        print('Features saved!')

    def forward(self):
        """Main processing pipeline"""
        print('---------------- Phoneme Detection Procedure Commence -----------------')

        start_time = time()

        for audio_fpath, audio_fname in zip(self.stim_fpath_list, self.stim_fname_list):
            print(f'Processing audio: {audio_fname}')
            audio_fname_wo_wav = audio_fname.split('.')[0]
            result_path = os.path.join(self.result_dir, audio_fname_wo_wav)

            # Create output directories
            self.create_output_directories(result_path)

            # Core processing
            asr_result = self.asr(audio_fpath, audio_fname_wo_wav, result_path)
            df = self.phoneme_detect(audio_fpath, audio_fname_wo_wav, result_path)

            # Optional processing based on configuration
            if self.enable_visualization:
                self.visualize_phoneme_onsets(audio_fpath, df, result_path, audio_fname_wo_wav)

            if self.enable_segmentation:
                self.segment_audio_by_phonemes(audio_fpath, df, result_path, audio_fname_wo_wav)

            if self.enable_feature_extraction:
                spike_train_50hz, spike_train_64hz = self.convert_to_spike_train(df)
                self.save_features(spike_train_50hz, spike_train_64hz, result_path, audio_fname_wo_wav)

        # Print completion summary
        end_time = time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f'Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Run phoneme detection
    phoneme_detection = PhonemeDetection(config)
    phoneme_detection.forward()