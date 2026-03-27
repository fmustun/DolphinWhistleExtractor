#!/usr/bin/env python3
"""
Inference pipeline diagnostic tool.

Generates spectrogram images from a WAV/FLAC recording, validates the
preprocessing pipeline, and analyses the distribution of model confidence
scores to help choose the right detection threshold.

Usage
-----
    # Inspect one recording (saves first N spectrogram images)
    python debug_inference.py --wav /path/to/recording.wav --model models/model_vgg_best.h5

    # Compare with a training image
    python debug_inference.py --wav /path/to/recording.wav \\
        --model models/model_vgg_best.h5 \\
        --training_jpg /media/DOLPHIN1/.../spectrograms/positives/whistle_000001.jpg

    # Analyse score distribution over 10 minutes to find the right threshold
    python debug_inference.py --wav /path/to/recording.wav \\
        --model models/model_vgg_best.h5 --score_analysis --analysis_duration 600

    # Disable resampling (if recordings are already at training sample rate)
    python debug_inference.py --wav /path/to/recording.wav --target_fs 0
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal.windows import blackman

# Use CPU for testing
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ── Spectrogram generation (mirrors predict_and_extract_online_new.py) ────────

def make_spectrogram(
    audio: np.ndarray,
    fs: int,
    start_sample: int,
    wlen: int = 1024,
    nfft: int = 1024,
    cut_low_frequency: float = 2.0,   # kHz
    cut_high_frequency: float = 22.0,  # kHz
) -> np.ndarray:
    """Return a uint8 (H, W) spectrogram array — before any resize."""
    hop = round(0.5 * wlen)
    spw = round(0.4 * fs)
    win_func = blackman(wlen, sym=False)

    x_w = audio[start_sample:start_sample + spw].astype(np.float32)
    f, _, sxx = scipy_spectrogram(
        x_w, fs,
        nperseg=wlen, noverlap=wlen - hop, nfft=nfft,
        window=win_func, scaling='density', mode='psd',
    )
    sxx = 10.0 * np.log10(np.abs(sxx) + 1e-19)
    sxx = (sxx - sxx.min()) / (sxx.max() - sxx.min() + 1e-12) * 255.0

    lo_hz = cut_low_frequency * 1000
    hi_hz = cut_high_frequency * 1000
    lo_i = int(np.searchsorted(f, lo_hz))
    hi_i = int(np.searchsorted(f, hi_hz))

    sxx_crop = np.flipud(sxx[lo_i:hi_i, :])
    return np.clip(sxx_crop, 0, 255).astype(np.uint8)


def load_audio(path: str, target_fs: int = None):
    ext = Path(path).suffix.lower()
    if ext == '.flac':
        try:
            import soundfile as sf
            x, fs = sf.read(path, always_2d=False)
        except ImportError:
            import librosa
            x, fs = librosa.load(path, sr=None, mono=True)
    else:
        fs, x = wavfile.read(path)

    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32)

    if target_fs and fs != target_fs:
        import librosa
        print(f"  Resampling {fs} Hz → {target_fs} Hz")
        x = librosa.resample(x, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return fs, x


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise and validate the inference spectrogram pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--wav', required=True,
                        help='WAV/FLAC recording to inspect')
    parser.add_argument('--model', default='models/model_vgg_best.h5',
                        help='Keras model (.h5 or .keras)')
    parser.add_argument('--training_jpg', default=None,
                        help='A JPEG from the training set for visual comparison')
    parser.add_argument('--out_dir', default='debug_spectrograms',
                        help='Folder to write diagnostic images')
    parser.add_argument('--n_windows', type=int, default=5,
                        help='Number of 0.4 s windows to inspect')
    parser.add_argument('--start_time', type=float, default=0.0,
                        help='Start time (seconds) in the recording')
    parser.add_argument('--CLF', type=float, default=2.0, help='Cut low freq (kHz)')
    parser.add_argument('--CHF', type=float, default=22.0, help='Cut high freq (kHz)')
    parser.add_argument('--wlen', type=int, default=1024, help='FFT window length')
    parser.add_argument('--nfft', type=int, default=1024, help='FFT points')
    parser.add_argument('--target_fs', type=int, default=96_000,
                        help='Resample to this rate (0 = no resampling)')
    parser.add_argument('--score_analysis', action='store_true',
                        help='Run score distribution analysis over a longer segment '
                             'to help find the right detection threshold')
    parser.add_argument('--analysis_duration', type=float, default=300,
                        help='Duration (seconds) to analyse for --score_analysis')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    target_fs = args.target_fs if args.target_fs > 0 else None

    # ── 1. Load audio ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Recording : {args.wav}")
    fs_orig, x = load_audio(args.wav, target_fs=target_fs)
    spw = round(0.4 * fs_orig)
    total_windows = (len(x) - int(args.start_time * fs_orig)) // spw
    print(f"Sample rate (after resampling): {fs_orig} Hz")
    print(f"Duration   : {len(x)/fs_orig:.1f} s  ({total_windows} × 0.4 s windows)")
    print(f"FFT freq resolution: {fs_orig/args.nfft:.1f} Hz/bin")
    print(f"Samples per 0.4 s window: {spw}")

    # ── 2. Generate N inference spectrograms ──────────────────────────────────
    print(f"\nGenerating {args.n_windows} spectrogram windows …")
    start_sample = int(args.start_time * fs_orig)

    model = None
    if os.path.exists(args.model):
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        print(f"Loading model: {args.model}")
        model = tf.keras.models.load_model(args.model, compile=False)
        model.trainable = False
        img_h, img_w = int(model.input_shape[1]), int(model.input_shape[2])
        print(f"Model input: {model.input_shape}")
    else:
        print(f"[WARNING] Model not found at '{args.model}' — skipping predictions")
        img_h, img_w = 224, 224

    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

    for i in range(min(args.n_windows, total_windows)):
        s0 = start_sample + i * spw
        gray = make_spectrogram(
            x, fs_orig, s0,
            wlen=args.wlen, nfft=args.nfft,
            cut_low_frequency=args.CLF,
            cut_high_frequency=args.CHF,
        )
        t_s = s0 / fs_orig

        print(f"\n  Window {i}: t={t_s:.2f}s  raw spectrogram shape={gray.shape}")

        # Save raw (before model resize)
        raw_path = os.path.join(args.out_dir, f"window_{i:02d}_raw.jpg")
        cv2.imwrite(raw_path, gray)

        # Resize to model input
        resized = cv2.resize(gray, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        rgb = np.stack([resized, resized, resized], axis=2)

        # Save resized (before VGG preprocessing)
        resized_path = os.path.join(args.out_dir, f"window_{i:02d}_resized_224.jpg")
        cv2.imwrite(resized_path, rgb)

        # Run model prediction
        if model is not None:
            model_input = vgg16_preprocess(rgb.astype(np.float32))[np.newaxis]
            pred = model.predict(model_input, verbose=0)[0]
            if len(pred) >= 2:
                print(f"    Prediction → class0={pred[0]:.4f}  class1={pred[1]:.4f}  "
                      f"{'POSITIVE ✓' if pred[1] >= 0.5 else 'negative'}")
            else:
                print(f"    Prediction → score={pred[0]:.4f}  "
                      f"{'POSITIVE ✓' if pred[0] >= 0.5 else 'negative'}")

    # ── 3. Compare with a training JPEG (if provided) ─────────────────────────
    if args.training_jpg and os.path.exists(args.training_jpg):
        print(f"\n{'─'*60}")
        print(f"Training JPEG: {args.training_jpg}")
        train_img = cv2.imread(args.training_jpg)
        if train_img is not None:
            print(f"  Native size (H×W): {train_img.shape[:2]}")
            # Resize the training JPEG the same way Keras would
            train_resized = cv2.resize(
                train_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST
            )
            train_path = os.path.join(args.out_dir, "training_image_resized_224.jpg")
            cv2.imwrite(train_path, train_resized)

            if model is not None:
                # Convert BGR (cv2) → RGB before VGG preprocess
                train_rgb = cv2.cvtColor(train_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
                model_input = vgg16_preprocess(train_rgb)[np.newaxis]
                pred = model.predict(model_input, verbose=0)[0]
                if len(pred) >= 2:
                    print(f"  Training image prediction → "
                          f"class0={pred[0]:.4f}  class1={pred[1]:.4f}  "
                          f"({'POSITIVE ✓' if pred[1] >= 0.5 else 'negative'})")
                else:
                    print(f"  Training image prediction → score={pred[0]:.4f}")
            print(f"  Saved to: {train_path}")
        else:
            print("  [ERROR] Could not read training JPEG")

    # ── 4. Score distribution analysis ────────────────────────────────────────
    if args.score_analysis and model is not None:
        print(f"\n{'='*60}")
        print(f"Score distribution analysis  (first {args.analysis_duration:.0f} s)")

        analysis_start = int(args.start_time * fs_orig)
        analysis_end   = min(len(x), analysis_start + int(args.analysis_duration * fs_orig))
        analysis_spw   = round(0.4 * fs_orig)
        n_analysis     = (analysis_end - analysis_start) // analysis_spw

        all_scores = []
        batch_size = 32
        imgs_batch = []

        print(f"  Processing {n_analysis} windows …", flush=True)
        for i in range(n_analysis):
            s0   = analysis_start + i * analysis_spw
            gray = make_spectrogram(x, fs_orig, s0,
                                    wlen=args.wlen, nfft=args.nfft,
                                    cut_low_frequency=args.CLF,
                                    cut_high_frequency=args.CHF)
            resized = cv2.resize(gray, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            rgb     = np.stack([resized, resized, resized], axis=2)
            imgs_batch.append(rgb)

            if len(imgs_batch) == batch_size or i == n_analysis - 1:
                arr   = np.array(imgs_batch, dtype=np.uint8)
                inp   = vgg16_preprocess(arr.astype(np.float32))
                preds = model.predict(inp, verbose=0)
                scores = preds[:, 1] if preds.shape[1] >= 2 else preds[:, 0]
                all_scores.extend(scores.tolist())
                imgs_batch = []

        all_scores = np.array(all_scores)
        print(f"\n  Score statistics over {len(all_scores)} windows:")
        print(f"    min    = {all_scores.min():.4f}")
        print(f"    max    = {all_scores.max():.4f}")
        print(f"    mean   = {all_scores.mean():.4f}")
        print(f"    median = {np.median(all_scores):.4f}")
        print(f"    p90    = {np.percentile(all_scores, 90):.4f}")
        print(f"    p95    = {np.percentile(all_scores, 95):.4f}")
        print(f"    p99    = {np.percentile(all_scores, 99):.4f}")
        print()

        thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        print(f"  Detections per threshold (this segment only):")
        print(f"  {'Threshold':>10}  {'Detections':>12}  {'Det. rate':>10}")
        print(f"  {'-'*36}")
        for thr in thresholds:
            n_det  = int((all_scores >= thr).sum())
            rate   = n_det / len(all_scores) * 100
            flag   = "  ← try this" if rate < 5 and n_det > 0 else ""
            print(f"  {thr:>10.2f}  {n_det:>12d}  {rate:>9.1f} %{flag}")

        # Save score histogram as text
        hist_path = os.path.join(args.out_dir, "score_histogram.txt")
        bins = np.linspace(0, 1, 21)
        counts, edges = np.histogram(all_scores, bins=bins)
        with open(hist_path, 'w') as f:
            f.write("class1_score_range  count\n")
            for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
                bar = '#' * min(c, 60)
                f.write(f"[{lo:.2f}-{hi:.2f}]  {c:6d}  {bar}\n")
        print(f"\n  Histogram saved to: {hist_path}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Diagnostic images saved to: {args.out_dir}/")
    print("Open the images to visually compare:")
    print("  window_XX_raw.jpg          — raw spectrogram before resize")
    print("  window_XX_resized_224.jpg  — what the model actually sees")
    if args.training_jpg:
        print("  training_image_resized_224.jpg — training image (should look similar)")
    print()
    print("If 'window_XX_resized_224.jpg' looks very different from")
    print("'training_image_resized_224.jpg', there is a pipeline mismatch.")
    print()
    print("Common causes:")
    print("  • Model precision/threshold: run --score_analysis to find right threshold")
    print("  • Recording sample rate ≠ training sample rate → set --target_fs")
    print("  • Wrong CLF/CHF → check --CLF and --CHF match training")
    print("  • Model not yet retrained with new spectrogram parameters")


if __name__ == '__main__':
    main()
