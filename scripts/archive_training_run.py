#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    default_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser(
        description=(
            'Archive a training run by copying lightweight artifacts and '
            'linking or copying dataset directories.'
        )
    )
    parser.add_argument('--run-name', default='run3')
    parser.add_argument('--model-path', default='models/run3/model_vgg_best.pt')
    parser.add_argument('--reports-dir', default='reports/run3')
    parser.add_argument('--figs-dir', default='figs/run3')
    parser.add_argument(
        '--archive-dir',
        default=f'saved_runs/{default_stamp}_run3_manual_test',
        help='Output directory for the archived run.',
    )
    parser.add_argument(
        '--dataset-dir',
        action='append',
        default=[],
        metavar='LABEL=PATH',
        help='Dataset directory to attach to the archive. Can be passed multiple times.',
    )
    parser.add_argument(
        '--copy-datasets',
        action='store_true',
        help='Physically copy dataset directories instead of creating symlinks.',
    )
    parser.add_argument(
        '--report-file',
        action='append',
        default=[],
        metavar='FILENAME',
        help='Copy only the named report file(s). Defaults to all files in the reports dir.',
    )
    parser.add_argument(
        '--figure-file',
        action='append',
        default=[],
        metavar='FILENAME',
        help='Copy only the named figure file(s). Defaults to all files in the figs dir.',
    )
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_dataset_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if '=' not in spec:
            raise ValueError(
                f'Invalid --dataset-dir value {spec!r}. Expected LABEL=PATH.'
            )
        label, raw_path = spec.split('=', 1)
        label = label.strip()
        if not label:
            raise ValueError(f'Invalid dataset label in {spec!r}.')
        parsed.append((label, repo_path(raw_path.strip())))
    return parsed


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f'{label} not found: {path}')


def copy_tree_contents(src_dir: Path, dst_dir: Path, selected_names: list[str] | None = None) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    selected_name_set = set(selected_names or [])
    for src_path in sorted(src_dir.iterdir()):
        if selected_name_set and src_path.name not in selected_name_set:
            continue
        dst_path = dst_dir / src_path.name
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
        copied.append(src_path.name)
    missing = sorted(selected_name_set - set(copied))
    if missing:
        raise FileNotFoundError(
            f'Missing selected files in {src_dir}: {missing}'
        )
    return copied


def link_or_copy_dataset(src_dir: Path, dst_dir: Path, copy_datasets: bool) -> dict[str, str]:
    if copy_datasets:
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        mode = 'copied'
    else:
        if dst_dir.exists() or dst_dir.is_symlink():
            dst_dir.unlink()
        os.symlink(src_dir.resolve(), dst_dir)
        mode = 'symlinked'
    return {'mode': mode, 'source': str(src_dir.resolve()), 'target': str(dst_dir)}


def maybe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_readme(
    archive_dir: Path,
    run_name: str,
    model_relpath: str,
    report_files: list[str],
    fig_files: list[str],
    dataset_entries: dict[str, dict[str, str]],
    run_summary: dict | None,
) -> str:
    lines: list[str] = [
        f'# Archived Training Run: {run_name}',
        '',
        f'- Created: {datetime.now().isoformat(timespec="seconds")}',
        f'- Archive dir: `{archive_dir}`',
        f'- Model: `{model_relpath}`',
    ]

    if run_summary:
        best_epoch = run_summary.get('best_epoch')
        best_val_loss = run_summary.get('best_val_loss')
        if best_epoch is not None:
            lines.append(f'- Best epoch: `{best_epoch}`')
        if best_val_loss is not None:
            lines.append(f'- Best validation loss: `{best_val_loss}`')

    lines.extend(['', '## Reports'])
    if report_files:
        lines.extend([f'- `{name}`' for name in report_files])
    else:
        lines.append('- None found')

    lines.extend(['', '## Figures'])
    if fig_files:
        lines.extend([f'- `{name}`' for name in fig_files])
    else:
        lines.append('- None found')

    lines.extend(['', '## Datasets'])
    if dataset_entries:
        for label, entry in dataset_entries.items():
            lines.append(
                f"- `{label}`: {entry['mode']} -> `{entry['source']}`"
            )
    else:
        lines.append('- None attached')

    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()

    archive_dir = repo_path(args.archive_dir)
    model_path = repo_path(args.model_path)
    reports_dir = repo_path(args.reports_dir)
    figs_dir = repo_path(args.figs_dir)
    datasets = parse_dataset_specs(args.dataset_dir)

    ensure_exists(model_path, 'Model checkpoint')
    ensure_exists(reports_dir, 'Reports directory')
    ensure_exists(figs_dir, 'Figures directory')
    for label, dataset_dir in datasets:
        ensure_exists(dataset_dir, f'Dataset {label!r}')

    model_archive_dir = archive_dir / 'model'
    reports_archive_dir = archive_dir / 'reports'
    figs_archive_dir = archive_dir / 'figs'
    datasets_archive_dir = archive_dir / 'datasets'

    model_archive_dir.mkdir(parents=True, exist_ok=True)
    copied_model_path = model_archive_dir / model_path.name
    shutil.copy2(model_path, copied_model_path)

    copied_report_files = copy_tree_contents(
        reports_dir,
        reports_archive_dir,
        selected_names=args.report_file or None,
    )
    copied_fig_files = copy_tree_contents(
        figs_dir,
        figs_archive_dir,
        selected_names=args.figure_file or None,
    )

    dataset_entries: dict[str, dict[str, str]] = {}
    datasets_archive_dir.mkdir(parents=True, exist_ok=True)
    for label, dataset_dir in datasets:
        target = datasets_archive_dir / label
        dataset_entries[label] = link_or_copy_dataset(
            dataset_dir,
            target,
            copy_datasets=args.copy_datasets,
        )

    run_summary_path = reports_archive_dir / 'run_summary.json'
    run_summary = maybe_load_json(run_summary_path)

    manifest = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'run_name': args.run_name,
        'archive_dir': str(archive_dir.resolve()),
        'model': {
            'source': str(model_path.resolve()),
            'archived_copy': str(copied_model_path.resolve()),
            'size_bytes': copied_model_path.stat().st_size,
        },
        'reports': {
            'source': str(reports_dir.resolve()),
            'archived_dir': str(reports_archive_dir.resolve()),
            'files': copied_report_files,
        },
        'figures': {
            'source': str(figs_dir.resolve()),
            'archived_dir': str(figs_archive_dir.resolve()),
            'files': copied_fig_files,
        },
        'datasets': dataset_entries,
        'run_summary': run_summary,
    }

    manifest_path = archive_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    readme_path = archive_dir / 'README.md'
    readme_path.write_text(
        build_readme(
            archive_dir=archive_dir,
            run_name=args.run_name,
            model_relpath=str(copied_model_path.relative_to(archive_dir)),
            report_files=copied_report_files,
            fig_files=copied_fig_files,
            dataset_entries=dataset_entries,
            run_summary=run_summary,
        )
    )

    print(f'Archive created: {archive_dir}')
    print(f'Model copy: {copied_model_path}')
    print(f'Reports copied: {len(copied_report_files)}')
    print(f'Figures copied: {len(copied_fig_files)}')
    print(f'Datasets attached: {len(dataset_entries)}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
