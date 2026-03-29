import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal.windows import blackman


def make_spectrogram_image(
    file_path,
    output_path,
    wlen=1024,
    nfft=1024,
    cut_low_frequency=2,
    cut_high_frequency=22,
    target_width_px=1167,
    target_height_px=875,
):
    fs, x = wavfile.read(file_path)

    if x.ndim > 1:
        x = x[:, 0]

    x = x.astype(np.float32)

    hop = round(0.5 * wlen)
    win = blackman(wlen, sym=False)

    f, t, sxx = spectrogram(
        x,
        fs,
        nperseg=wlen,
        noverlap=wlen - hop,
        nfft=nfft,
        window=win,
        scaling="density",
        mode="psd",
    )

    sxx = 10 * np.log10(np.abs(sxx) + 1e-19)
    sxx = (sxx - np.min(sxx)) / (np.max(sxx) - np.min(sxx) + 1e-12) * 255.0

    low_freq_hz = cut_low_frequency * 1000
    high_freq_hz = cut_high_frequency * 1000
    low_idx = np.searchsorted(f, low_freq_hz)
    high_idx = np.searchsorted(f, high_freq_hz)

    sxx_cropped = sxx[low_idx:high_idx, :]
    sxx_cropped = np.flipud(sxx_cropped)
    img_gray = np.clip(sxx_cropped, 0, 255).astype(np.uint8)

    resized = cv2.resize(
        img_gray,
        (target_width_px, target_height_px),
        interpolation=cv2.INTER_NEAREST,
    )
    image = np.stack([resized, resized, resized], axis=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def iter_audio_files(input_dir):
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(".wav"):
                yield Path(root) / name


def main():
    parser = argparse.ArgumentParser(
        description="Create spectrogram images for 0.4 s WAV clips."
    )
    parser.add_argument(
        "--input_dir",
        default="/media/DOLPHIN1/Wh_detection_database_NEURIPS/all_years_100k_dataset/wav_files",
        help="Folder containing WAV clips.",
    )
    parser.add_argument(
        "--output_dir",
        default="/media/DOLPHIN1/Wh_detection_database_NEURIPS/all_years_100k_dataset/spectrograms",
        help="Folder where spectrogram JPGs will be written.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    audio_files = list(iter_audio_files(input_dir))
    if not audio_files:
        print(f"No WAV files found in: {input_dir}")
        return

    skipped = 0
    print(f"Found {len(audio_files)} WAV files.")
    for idx, wav_path in enumerate(audio_files, start=1):
        stem = wav_path.stem.lower()
        if stem.startswith("whistle"):
            label_dir = output_dir / "positives"
        elif stem.startswith("noise"):
            label_dir = output_dir / "negatives"
        else:
            skipped += 1
            continue

        out_path = (label_dir / wav_path.name).with_suffix(".jpg")

        try:
            make_spectrogram_image(wav_path, out_path)
        except Exception as exc:
            print(f"[{idx}/{len(audio_files)}] Failed: {wav_path} -> {exc}")
            continue

        if idx % 500 == 0 or idx == len(audio_files):
            print(f"[{idx}/{len(audio_files)}] Processed")

    if skipped:
        print(f"Skipped {skipped} file(s) whose names don't start with 'whistle' or 'noise'.")
    print(f"Done. Spectrograms saved to: {output_dir}")


if __name__ == "__main__":
    main()
