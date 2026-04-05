"""
Dolphin whistle detection — inference engine  (new model edition)

Processes WAV/FLAC recordings with the retrained VGG16-based model.

Spectrogram parameters are identical to those used during training
(batch_plot_0p4s_spectrograms.py):
    wlen = 1024, nfft = 1024, hop = 512  (50 % overlap)
    CLF = 2 kHz, CHF = 22 kHz
    0.4 s sliding window, scipy PSD mode, 10·log10, min-max [0-255],
    flipud → INTER_NEAREST resize to model input size,
    backend-specific model normalization before inference.

Set `target_fs` in ProcessingConfig (default 96 000 Hz) to match the sample
rate used when creating the training spectrograms.  If recordings were captured
at a different rate they are resampled automatically.

Key differences vs predict_and_extract_online.py
-------------------------------------------------
* Spectrogram generation is self-contained (no dependency on utils.py)
* Audio window = 0.4 s  (was hardcoded to 0.8 s in utils.process_audio_file)
* hop = round(0.5 * wlen)  (was round(0.8 * wlen))
* wlen / nfft = 1024  (were 2048)
* CLF = 2 kHz, CHF = 22 kHz  (were 3 / 20)
* backend-specific model normalization is always applied
* Binary threshold applied consistently for all output shapes
"""

import io
import logging
import mmap
import os
import sys
import time
import warnings
import concurrent.futures
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.whistle_torch import (  # noqa: E402
    ARCHITECTURE_LEGACY,
    DEFAULT_NORMALIZATION_MEAN,
    DEFAULT_NORMALIZATION_STD,
    SpectrogramConfig,
    build_torch_model,
    make_spectrogram_batch as shared_make_spectrogram_batch,
    normalize_uint8_batch_to_torch,
    resample_audio_if_needed,
)

# ── Logging & warnings ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@lru_cache(maxsize=1)
def get_tf_module():
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess  # type: ignore
    except ImportError as exc:
        raise ImportError(
            'TensorFlow is required only for .h5/.keras inference. '
            'Install tensorflow or use a .pt model.'
        ) from exc

    for dev in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(dev, True)
        except Exception:
            pass

    return tf, vgg16_preprocess


# ── Configuration dataclass ───────────────────────────────────────────────────
@dataclass
class ProcessingConfig:
    """
    All parameters controlling inference.

    Spectrogram defaults match batch_plot_0p4s_spectrograms.py exactly.
    image_size is derived from model.input_shape at runtime.
    """
    batch_size: int
    cut_low_frequency: float = 2.0    # kHz
    cut_high_frequency: float = 22.0  # kHz
    save_positive_examples: bool = False
    binary_threshold: float = 0.5
    image_size: Tuple[int, int] = (224, 224)  # (height, width) — set from model
    # Spectrogram parameters — keep in sync with training
    wlen: int = 1024
    nfft: int = 1024
    sliding_window: float = 0.4   # seconds per clip
    # Audio resampling — set to the sample rate used during training.
    # If None, no resampling is applied (only safe when recordings are
    # guaranteed to be at the same rate as the training data).
    target_fs: Optional[int] = 96_000

    @property
    def hop(self) -> int:
        return round(0.5 * self.wlen)

    @property
    def batch_duration(self) -> float:
        return self.batch_size * self.sliding_window


@dataclass
class InferenceModel:
    backend: str
    model: object
    image_size: Tuple[int, int]
    output_type: str
    device: Optional[torch.device] = None
    normalization_mean: Tuple[float, float, float] = DEFAULT_NORMALIZATION_MEAN
    normalization_std: Tuple[float, float, float] = DEFAULT_NORMALIZATION_STD
    architecture: str = ARCHITECTURE_LEGACY
    spectrogram_config: Optional[dict] = None


# ── Spectrogram generation ────────────────────────────────────────────────────
def make_spectrogram_batch(
    audio: np.ndarray,
    fs: int,
    start_sample: int,
    n_windows: int,
    config: ProcessingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spectrogram images for up to `n_windows` consecutive 0.4 s clips.

    The pipeline matches training exactly:
      raw audio (resampled to target_fs if needed)
      → scipy PSD spectrogram → 10·log10 → min-max [0-255]
      → freq crop → flipud
      → INTER_NEAREST resize to model input size
      → stack RGB → model-specific normalization

    Parameters
    ----------
    audio        : full mono float32 waveform
    fs           : sample rate (Hz)
    start_sample : first sample of this batch
    n_windows    : maximum number of 0.4 s clips to process
    config       : ProcessingConfig

    Returns
    -------
    imgs_uint8   : (N, H, W, 3) uint8  — for optional disk saving
    start_sec    : (N,) float64 — window start time in seconds from file start
    """
    spectrogram_config = SpectrogramConfig(
        image_size=config.image_size,
        cut_low_frequency=config.cut_low_frequency,
        cut_high_frequency=config.cut_high_frequency,
        wlen=config.wlen,
        nfft=config.nfft,
        sliding_window=config.sliding_window,
        target_fs=config.target_fs,
    )
    return shared_make_spectrogram_batch(
        audio,
        fs,
        start_sample,
        n_windows,
        spectrogram_config,
    )


def prepare_inputs_for_backend(
    images_uint8: np.ndarray,
    inference_model: InferenceModel,
) -> object:
    if inference_model.backend == 'tensorflow':
        tf, vgg16_preprocess = get_tf_module()
        model_input = vgg16_preprocess(images_uint8.astype(np.float32))
        return tf.convert_to_tensor(model_input, dtype=tf.float32)

    return normalize_uint8_batch_to_torch(
        images_uint8,
        mean=inference_model.normalization_mean,
        std=inference_model.normalization_std,
        device=inference_model.device,
    )


def predict_batch_tf(model: object, images: object) -> np.ndarray:
    return model(images, training=False).numpy()


def predict_batch(
    inference_model: InferenceModel,
    images_uint8: np.ndarray,
) -> np.ndarray:
    model_input = prepare_inputs_for_backend(images_uint8, inference_model)
    if inference_model.backend == 'tensorflow':
        return predict_batch_tf(inference_model.model, model_input)

    with torch.no_grad():
        logits = inference_model.model(model_input)
        return torch.softmax(logits, dim=1).cpu().numpy()


def detect_model_output_type(model: object) -> str:
    dummy = np.zeros((1,) + tuple(model.input_shape[1:]), dtype=np.float32)
    out = model.predict(dummy, verbose=0)
    if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
        return 'binary'
    if out.ndim == 2 and out.shape[1] >= 2:
        return 'categorical'
    return 'unknown'


def get_torch_device(cpu_only: bool) -> torch.device:
    if cpu_only or os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda')


def load_inference_model(model_path: str, cpu_only: bool = False) -> InferenceModel:
    logger.info(f"Loading model from {model_path}")
    suffix = Path(model_path).suffix.lower()

    if suffix in ('.h5', '.keras'):
        tf, _ = get_tf_module()
        model = tf.keras.models.load_model(model_path, compile=False)
        model.trainable = False
        model_type = detect_model_output_type(model)
        logger.info(f"Model type: {model_type}  |  Input: {model.input_shape}")
        _ = model.predict(
            np.zeros((1,) + tuple(model.input_shape[1:]), dtype=np.float32),
            verbose=0,
        )
        return InferenceModel(
            backend='tensorflow',
            model=model,
            image_size=(int(model.input_shape[1]), int(model.input_shape[2])),
            output_type=model_type,
        )

    if suffix == '.pt':
        device = get_torch_device(cpu_only)
        checkpoint = torch.load(model_path, map_location=device)
        architecture = checkpoint.get('architecture', ARCHITECTURE_LEGACY)
        model = build_torch_model(
            architecture=architecture,
            pretrained_backbone=False,
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        img_size = int(checkpoint.get('img_size', 224))
        normalization = checkpoint.get('normalization', {})
        mean = tuple(normalization.get('mean', DEFAULT_NORMALIZATION_MEAN))
        std = tuple(normalization.get('std', DEFAULT_NORMALIZATION_STD))
        spectrogram_config = checkpoint.get('spectrogram_config')
        with torch.no_grad():
            dummy = torch.zeros((1, 3, img_size, img_size), device=device, dtype=torch.float32)
            _ = model(dummy)
        logger.info(f"Model type: categorical  |  Input: (None, {img_size}, {img_size}, 3)")
        logger.info(f"PyTorch inference device: {device}")
        return InferenceModel(
            backend='torch',
            model=model,
            image_size=(img_size, img_size),
            output_type='categorical',
            device=device,
            normalization_mean=mean,
            normalization_std=std,
            architecture=architecture,
            spectrogram_config=spectrogram_config,
        )

    raise ValueError(f"Unsupported model format: {suffix}. Expected .pt, .h5 or .keras")


def get_positive_scores(
    predictions: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (positive_indices, class-1 scores) for a prediction batch.

    Works for both binary (1 output) and categorical (≥2 outputs) models.
    The threshold is applied consistently to the class-1 probability.
    """
    scores = predictions[:, 1] if predictions.shape[1] >= 2 else predictions[:, 0]
    return np.where(scores >= threshold)[0], scores


# ── I/O helpers ───────────────────────────────────────────────────────────────
def save_detection_image(
    image_uint8: np.ndarray,
    t_start: float,
    t_end: float,
    folder: Path,
) -> None:
    dest = folder / "positive"
    dest.mkdir(exist_ok=True)
    cv2.imwrite(str(dest / f"{t_start:.2f}-{t_end:.2f}.jpg"), image_uint8)


def save_detections_csv(
    record_names: List[str],
    start_times: List[float],
    end_times: List[float],
    scores: List[float],
    path: str,
) -> None:
    pd.DataFrame({
        'file_name': record_names,
        'initial_point': start_times,
        'finish_point': end_times,
        'confidence': scores,
    }).to_csv(path, index=False)


def load_audio(file_path: str) -> Tuple[int, np.ndarray]:
    """
    Load a WAV or FLAC file and return (fs, mono float32 array).
    Uses memory mapping for WAV files to reduce I/O overhead.
    """
    ext = Path(file_path).suffix.lower()
    if ext == '.flac':
        try:
            import soundfile as sf
            x, fs = sf.read(file_path, always_2d=False)
        except ImportError:
            try:
                import librosa
                x, fs = librosa.load(file_path, sr=None, mono=True)
            except ImportError:
                raise ImportError(
                    "Install soundfile (`pip install soundfile`) or "
                    "librosa to process FLAC files."
                )
    else:
        with open(file_path, 'rb') as fid:
            mm = mmap.mmap(fid.fileno(), 0, access=mmap.ACCESS_READ)
            fs, x = wavfile.read(io.BytesIO(mm))

    if x.ndim > 1:
        x = x[:, 0]
    return fs, x.astype(np.float32)


# ── Per-file processing ───────────────────────────────────────────────────────
def process_and_predict(
    file_path: str,
    config: ProcessingConfig,
    start_time: float,
    end_time: Optional[float],
    model: InferenceModel,
    saving_folder: Path,
) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Slide a 0.4 s window over one audio file, generate spectrograms,
    run model predictions, and return all positive detections.
    """
    t0 = time.time()
    file_name = Path(file_path).name
    record_names: List[str] = []
    starts: List[float] = []
    ends: List[float] = []
    scores_out: List[float] = []

    try:
        fs, x = load_audio(file_path)

        # Resample to the target sample rate used during training
        original_fs = fs
        fs, x = resample_audio_if_needed(x, fs, config.target_fs)
        if fs != original_fs:
            logger.info(f"{file_name}: resampling {original_fs} Hz → {fs} Hz")

        logger.info(
            f"{file_name}: fs={fs} Hz, "
            f"spw={round(config.sliding_window * fs)} samples/window"
        )

        N = min(int(end_time * fs), len(x)) if end_time is not None else len(x)
        start_sample = int(start_time * fs)
        spw = round(config.sliding_window * fs)
        total_windows = max(0, (N - start_sample) // spw)

        if total_windows == 0:
            return record_names, starts, ends, scores_out

        num_batches = int(np.ceil(total_windows / config.batch_size))

        for b in tqdm(range(num_batches), desc=file_name, leave=False, colour='blue'):
            batch_start = start_sample + b * config.batch_size * spw
            imgs_uint8, win_starts = make_spectrogram_batch(
                x, fs, batch_start, config.batch_size, config,
            )

            if imgs_uint8.shape[0] == 0:
                continue

            preds = predict_batch(model, imgs_uint8)

            pos_idx, all_scores = get_positive_scores(preds, config.binary_threshold)

            for idx in pos_idx:
                t_s = round(float(win_starts[idx]), 2)
                t_e = round(t_s + config.sliding_window, 2)
                record_names.append(file_name)
                starts.append(t_s)
                ends.append(t_e)
                scores_out.append(float(all_scores[idx]))

                if config.save_positive_examples:
                    save_detection_image(imgs_uint8[idx], t_s, t_e, saving_folder)

            del imgs_uint8, preds

        logger.info(
            f"{file_name}: {len(record_names)} detections in {time.time()-t0:.1f}s"
        )

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

    return record_names, starts, ends, scores_out


def process_single_file(
    file_name: str,
    recording_folder: Path,
    saving_folder: Path,
    config: ProcessingConfig,
    start_time: float,
    end_time: Optional[float],
    model: InferenceModel,
    pbar: tqdm,
) -> None:
    """Wrapper that handles one WAV/FLAC file and writes the predictions CSV."""
    try:
        if not file_name.lower().endswith(('.wav', '.flac')):
            return

        file_stem = Path(file_name).stem
        file_path = recording_folder / file_name
        file_saving_folder = saving_folder / file_stem
        prediction_path = file_saving_folder / f"{file_stem}.wav_predictions.csv"

        if prediction_path.exists():
            logger.info(f"Skipping {file_name}: already processed")
            return

        logger.info(f"Processing: {file_stem}")

        file_saving_folder.mkdir(exist_ok=True, parents=True)
        record_names, starts, ends, scores = process_and_predict(
            str(file_path), config, start_time, end_time, model, file_saving_folder,
        )

        save_detections_csv(record_names, starts, ends, scores, str(prediction_path))
        logger.info(f"Saved {len(record_names)} detections for {file_name}")

    except Exception as e:
        logger.error(f"Error in {file_name}: {e}")
    finally:
        pbar.update(1)


# ── Resource heuristics ───────────────────────────────────────────────────────
def get_optimal_batch_size(file_count: int) -> int:
    try:
        import psutil
        mem_gb = psutil.virtual_memory().available / 1024 ** 3
    except ImportError:
        mem_gb = 8
    gpu_ok = bool(torch.cuda.is_available())
    if gpu_ok:
        return 64 if mem_gb > 16 else (32 if mem_gb > 8 else 16)
    return 128 if mem_gb > 32 else (64 if mem_gb > 16 else (32 if mem_gb > 8 else 16))


def get_optimal_worker_count(file_count: int) -> int:
    try:
        import psutil
        mem_gb = psutil.virtual_memory().available / 1024 ** 3
    except ImportError:
        mem_gb = 8
    cpu = os.cpu_count() or 8
    if file_count < 5:
        return max(1, min(file_count, cpu // 2))
    if mem_gb < 4:
        return max(1, cpu // 4)
    if mem_gb < 8:
        return max(1, cpu // 2)
    return max(1, int(cpu * 0.75))


# ── Main entry point ─────────────────────────────────────────────────────────
def process_predict_extract(
    recording_folder_path: str,
    saving_folder: str,
    cut_low_freq: float = 2.0,
    cut_high_freq: float = 22.0,
    start_time: float = 0,
    end_time: Optional[float] = None,
    batch_size: int = 64,
    save_positives: bool = True,
    model_path: str = "models/model_vgg_best.h5",
    binary_threshold: float = 0.5,
    max_workers: Optional[int] = None,
    specific_files: Optional[List[str]] = None,
    target_fs: Optional[int] = 96_000,
    cpu_only: bool = False,
) -> None:
    """
    Load model, iterate over recordings, run whistle detection, write CSV results.

    Parameters
    ----------
    recording_folder_path : folder containing WAV/FLAC recordings
    saving_folder         : output root; one sub-folder is created per recording
    cut_low_freq          : lower frequency bound for spectrograms (kHz)
    cut_high_freq         : upper frequency bound for spectrograms (kHz)
    start_time            : skip this many seconds at the start of each file
    end_time              : stop processing at this second (None = full file)
    batch_size            : number of 0.4 s windows per inference batch
    save_positives        : if True, save spectrogram images for positive detections
    model_path            : path to the saved model (.pt, .h5 or .keras)
    binary_threshold      : detection threshold applied to class-1 probability
    max_workers           : thread-pool size (None = auto)
    specific_files        : list of filenames (relative to recording_folder_path)
                            to process instead of the whole folder
    """
    recording_folder = Path(recording_folder_path)
    out_folder = Path(saving_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    t_total = time.time()

    # ── Determine file list ───────────────────────────────────────────────────
    audio_exts = ('.wav', '.flac')
    if specific_files:
        files = sorted([
            f for f in specific_files
            if (recording_folder / f).exists()
            and Path(f).suffix.lower() in audio_exts
        ])
    else:
        files = sorted([
            f for f in os.listdir(recording_folder_path)
            if Path(f).suffix.lower() in audio_exts
        ])

    if not files:
        logger.info("No files to process.")
        return

    logger.info(f"Found {len(files)} files to process")

    if batch_size <= 0:
        batch_size = get_optimal_batch_size(len(files))
        logger.info(f"Auto batch size: {batch_size}")
    if max_workers is None or max_workers <= 0:
        max_workers = get_optimal_worker_count(len(files))
        logger.info(f"Auto worker count: {max_workers}")

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = load_inference_model(model_path, cpu_only=cpu_only)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    img_h, img_w = model.image_size

    config = ProcessingConfig(
        batch_size=batch_size,
        cut_low_frequency=cut_low_freq,
        cut_high_frequency=cut_high_freq,
        save_positive_examples=save_positives,
        binary_threshold=binary_threshold,
        image_size=(img_h, img_w),
        target_fs=target_fs,
    )

    checkpoint_spectrogram_config = SpectrogramConfig.from_metadata(model.spectrogram_config)
    mismatches = []
    requested = {
        'image_size': tuple(config.image_size),
        'cut_low_frequency': float(config.cut_low_frequency),
        'cut_high_frequency': float(config.cut_high_frequency),
        'wlen': int(config.wlen),
        'nfft': int(config.nfft),
        'sliding_window': float(config.sliding_window),
        'target_fs': config.target_fs,
    }
    trained = checkpoint_spectrogram_config.to_metadata()
    for key, requested_value in requested.items():
        trained_value = trained.get(key)
        if key == 'image_size':
            trained_value = tuple(trained_value) if trained_value is not None else None
        if trained_value != requested_value:
            mismatches.append(f'{key}: train={trained_value} inference={requested_value}')
    if mismatches:
        logger.warning(
            'Inference settings differ from the Torch training checkpoint: '
            + '; '.join(mismatches)
        )

    # ── Process files in parallel ─────────────────────────────────────────────
    with tqdm(total=len(files), desc="Files", position=0, leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    process_single_file,
                    fname, recording_folder, out_folder,
                    config, start_time, end_time, model, pbar,
                )
                for fname in files
            ]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"Thread error: {e}")

    logger.info(f"Done. {len(files)} files processed in {time.time()-t_total:.1f}s")
