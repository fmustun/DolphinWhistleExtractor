#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


AUDIO_EXTENSIONS = {'.wav', '.flac'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create chunked file lists for Jean Zay inference arrays.'
    )
    parser.add_argument(
        '--recordings-dir',
        required=True,
        help='Directory containing WAV/FLAC files to process.',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory where chunk text files and the manifest will be written.',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Number of recordings per chunk file.',
    )
    parser.add_argument(
        '--prefix',
        default='chunk',
        help='Chunk filename prefix.',
    )
    parser.add_argument(
        '--skip-existing-predictions-dir',
        default='',
        help=(
            'Optional inference output directory. If provided, recordings with an '
            'existing "<stem>/<stem>.wav_predictions.csv" file are skipped.'
        ),
    )
    return parser.parse_args()


def list_audio_files(recordings_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in recordings_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def has_prediction(output_dir: Path, file_name: str) -> bool:
    stem = Path(file_name).stem
    prediction_path = output_dir / stem / f'{stem}.wav_predictions.csv'
    return prediction_path.exists()


def write_chunks(file_names: list[str], output_dir: Path, prefix: str, chunk_size: int) -> list[str]:
    chunk_names: list[str] = []
    for chunk_index, start in enumerate(range(0, len(file_names), chunk_size)):
        chunk_file_names = file_names[start:start + chunk_size]
        chunk_name = f'{prefix}_{chunk_index:04d}.txt'
        chunk_path = output_dir / chunk_name
        chunk_path.write_text(''.join(f'{file_name}\n' for file_name in chunk_file_names))
        chunk_names.append(chunk_name)
    return chunk_names


def main() -> None:
    args = parse_args()

    recordings_dir = Path(args.recordings_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not recordings_dir.is_dir():
        raise FileNotFoundError(f'Recordings directory not found: {recordings_dir}')
    if args.chunk_size <= 0:
        raise ValueError('--chunk-size must be positive')

    file_names = list_audio_files(recordings_dir)
    skipped_existing = 0

    if args.skip_existing_predictions_dir:
        predictions_dir = Path(args.skip_existing_predictions_dir).expanduser().resolve()
        filtered_file_names: list[str] = []
        for file_name in file_names:
            if has_prediction(predictions_dir, file_name):
                skipped_existing += 1
                continue
            filtered_file_names.append(file_name)
        file_names = filtered_file_names

    chunk_names = write_chunks(file_names, output_dir, args.prefix, args.chunk_size)
    manifest = {
        'recordings_dir': str(recordings_dir),
        'output_dir': str(output_dir),
        'chunk_size': args.chunk_size,
        'prefix': args.prefix,
        'total_recordings': len(file_names),
        'chunk_count': len(chunk_names),
        'chunks': chunk_names,
        'skipped_existing_predictions': skipped_existing,
    }
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
