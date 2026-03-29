#!/usr/bin/env python3
"""
Dolphin Whistle Detection and Extraction Tool  (new model edition)

Uses the retrained VGG16-based model and updated spectrogram parameters
that match batch_plot_0p4s_spectrograms.py:
    wlen=1024, nfft=1024, hop=512, CLF=2 kHz, CHF=22 kHz,
    0.4 s sliding window, VGG16 preprocess_input.

Examples
--------
    # Basic run
    python main_new.py --recordings /path/to/wavs --saving_folder /path/to/out

    # Custom threshold, save positive spectrogram images
    python main_new.py --recordings /path/to/wavs --saving_folder /out \\
        --threshold 0.6 --save_p

    # Process only a listed subset of files
    python main_new.py --specific_files file_list.txt --save_p
"""

import argparse
import json
import os
import sys

from predict_and_extract_online import process_predict_extract


def read_file_list(file_path: str):
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except (IOError, FileNotFoundError) as e:
        print(f"Error reading file list '{file_path}': {e}")
        sys.exit(1)


def load_config(config_path: str) -> dict:
    """Load JSON config if present; otherwise create a template and return it."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from: {config_path}")
            return config
        except json.JSONDecodeError:
            print(f"Config file '{config_path}' is invalid JSON — using defaults.")
        except IOError as e:
            print(f"Error reading config '{config_path}': {e} — using defaults.")

    config = {"recordings": "", "saving_folder": ""}
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Default configuration saved to: {config_path}")
    except IOError as e:
        print(f"Could not save config to '{config_path}': {e}")
    return config


def main():
    config_path = os.path.expanduser("~/.predict_extract_config.json")
    config = load_config(config_path)

    # ── Defaults ──────────────────────────────────────────────────────────────
    default_model_path    = "models/model_vgg_best.h5"
    default_recordings    = config.get("recordings", "")
    default_saving_folder = config.get("saving_folder", "")
    default_batch_size    = 64
    default_max_workers   = 8
    default_CLF           = 2.0    # kHz — must match training
    default_CHF           = 22.0   # kHz — must match training
    default_threshold     = 0.5

    # ── Argument parser ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Detect dolphin whistles in audio recordings (new model).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument('--model_path', default=default_model_path,
                        help='Path to the trained Keras model (.h5 or .keras)')
    parser.add_argument('--recordings', default=default_recordings,
                        help='Folder containing WAV/FLAC recordings')
    parser.add_argument('--saving_folder', default=default_saving_folder,
                        help='Output folder for detection CSVs and images')

    # Time range
    parser.add_argument('--start_time', type=float, default=0,
                        help='Skip this many seconds at the start of each file')
    parser.add_argument('--end_time', type=float, default=None,
                        help='Stop processing at this second (None = full file)')

    # Inference
    parser.add_argument('--batch_size', type=int, default=default_batch_size,
                        help='Number of 0.4 s windows per inference batch')
    parser.add_argument('--max_workers', type=int, default=default_max_workers,
                        help='Parallel worker threads (0 = auto)')

    # Spectrogram frequency bounds — keep at training values unless retraining
    parser.add_argument('--CLF', type=float, default=default_CLF,
                        help='Cut low frequency (kHz)')
    parser.add_argument('--CHF', type=float, default=default_CHF,
                        help='Cut high frequency (kHz)')

    # Detection
    parser.add_argument('--threshold', type=float, default=default_threshold,
                        help='Class-1 probability threshold for positive detection')

    # File selection
    parser.add_argument('--specific_files',
                        help='Text file listing specific filenames to process '
                             '(one per line, relative to --recordings)')

    # Output options
    parser.add_argument('--save_p', action='store_true',
                        help='Save spectrogram images of positive detections to disk')

    parser.add_argument('--target_fs', type=int, default=96_000,
                        help='Resample recordings to this rate before computing '
                             'spectrograms (must match the rate used during training). '
                             'Set to 0 to disable resampling.')

    parser.add_argument('--cpu', action='store_true',
                        help='Disable GPU and run inference on CPU only')

    parser.add_argument('--version', action='version', version='%(prog)s 2.0.0')

    args = parser.parse_args()

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("CPU-only mode: GPU disabled.")

    # ── Validation ────────────────────────────────────────────────────────────
    if not os.path.exists(args.model_path):
        print(f"Error: model not found at '{args.model_path}'")
        sys.exit(1)

    if not args.recordings:
        print("Error: --recordings path is required.")
        sys.exit(1)

    if not os.path.exists(args.recordings):
        print(f"Error: recordings folder not found at '{args.recordings}'")
        sys.exit(1)

    if not args.saving_folder:
        print("Error: --saving_folder path is required.")
        sys.exit(1)

    os.makedirs(args.saving_folder, exist_ok=True)

    specific_files = None
    if args.specific_files:
        if not os.path.exists(args.specific_files):
            print(f"Error: file list not found at '{args.specific_files}'")
            sys.exit(1)
        specific_files = read_file_list(args.specific_files)
        print(f"Processing {len(specific_files)} files from '{args.specific_files}'")

    # Persist last-used paths for future runs
    config["recordings"]    = args.recordings
    config["saving_folder"] = args.saving_folder
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except IOError:
        pass

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"Starting whistle detection with {args.max_workers} workers…")
    try:
        process_predict_extract(
            recording_folder_path=args.recordings,
            saving_folder=args.saving_folder,
            cut_low_freq=args.CLF,
            cut_high_freq=args.CHF,
            start_time=args.start_time,
            end_time=args.end_time,
            batch_size=args.batch_size,
            save_positives=args.save_p,
            model_path=args.model_path,
            binary_threshold=args.threshold,
            max_workers=args.max_workers,
            specific_files=specific_files,
            target_fs=args.target_fs if args.target_fs > 0 else None,
        )
        print("Processing completed successfully.")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
