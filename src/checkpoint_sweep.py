"""Exploratory test metrics for every retained MAIN checkpoint.

This post-hoc report must not replace the validation-selected best.pt, the
official one-shot metrics.json, or an unbiased held-out test estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import trainer as run
from uncertainty_weighting import validate_checkpoint_weighting


def sweep(run_dir, output, device_override=None):
    directory, manifest = run._load_completed_run(run_dir)
    config = manifest['config']
    if config['variant'] != 'MAIN' or config['checkpoint_keep_last'] != 100:
        raise ValueError('The 100-epoch test sweep supports MAIN retention only')
    paths = manifest.get('retained_epoch_checkpoints')
    completed = manifest.get('completed_epochs')
    if not isinstance(completed, int) or completed < 1 or not isinstance(paths, list):
        raise ValueError('Missing retained-checkpoint inventory')
    expected_epochs = list(range(max(1, completed - 99), completed + 1))
    expected_paths = [f'checkpoints/epoch_{epoch:06d}.pt' for epoch in expected_epochs]
    if paths != expected_paths:
        raise ValueError('Retained-checkpoint inventory does not match the final 100 epochs')
    history = run.read_json(directory / 'history.json')
    if len(history) != completed or any(row.get('epoch') != index for index, row in enumerate(history, 1)):
        raise ValueError('Training history does not match completed epoch count')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Report already exists: {output}')
    prepared, metadata = run._prepared_for_run(directory, manifest)
    run.verify_artifact(prepared, metadata, 'test.npz')
    archive = directory / 'checkpoints'
    if archive.is_symlink() or archive.resolve().parent != directory:
        raise ValueError('Checkpoint archive must be inside the run directory')
    device = run.validate_device(device_override or config['device'], config['precision'])
    run.seed_everything(config['seed'])
    model = run.build_model(metadata, config).to(device)
    loader = run.make_loader(prepared / 'test.npz', config['batch_size'],
                             config['seed'], config['num_workers'])
    rows = []
    for epoch, name in zip(expected_epochs, paths):
        path = directory / name
        if path.is_symlink() or path.resolve().parent != (directory / 'checkpoints').resolve():
            raise ValueError(f'Checkpoint escapes its archive: {name}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('protocol', 'run_id', 'config_hash', 'code_hashes', 'prepared_hash'):
            if checkpoint.get(key) != manifest.get(key):
                raise ValueError(f'Checkpoint identity mismatch at epoch {epoch}: {key}')
        if (checkpoint.get('epoch') != epoch or checkpoint.get('selected_epoch') != epoch
                or checkpoint.get('config') != config or checkpoint.get('metadata') != metadata):
            raise ValueError(f'Checkpoint configuration or epoch mismatch: {name}')
        validate_checkpoint_weighting(checkpoint, config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        metrics, _ = run.predict(model, loader, metadata, device, config['precision'])
        if metrics['n_samples'] != metadata['counts']['test']:
            raise ValueError(f'Test row count mismatch: {name}')
        row = {'epoch': epoch, 'checkpoint_file': name,
               'validation': history[epoch - 1]['validation'],
               'checkpoint_sha256': run.sha256_file(path), 'test': metrics}
        rows.append(row)
        print(json.dumps({'epoch': epoch, 'test': metrics}, allow_nan=False), flush=True)
    # Cross-check the validation-selected checkpoint if it is among the retained epochs.
    official_path = directory / 'metrics.json'
    official = run.read_json(official_path) if official_path.exists() else None
    selected_epoch = manifest['selected_epoch']
    selected_row = next((row for row in rows if row['epoch'] == selected_epoch), None)
    if official is not None and selected_row is not None:
        if (official.get('checkpoint_sha256') != manifest['best_checkpoint_sha256']
                or official.get('selected_epoch') != selected_epoch
                or official.get('test') != selected_row['test']):
            raise ValueError('Retained selected checkpoint disagrees with official test metrics')
    ranked = sorted(rows, key=lambda row: (row['test']['mean_error_m'], row['epoch']))
    report = {
        'status': 'EXPLORATORY_POST_HOC_TEST_SWEEP',
        'interpretation': 'Minimum test error among evaluated epochs is optimistically biased; keep the validation-selected best.pt and official metrics.json for paper reporting.',
        'run_id': manifest['run_id'], 'config_hash': manifest['config_hash'],
        'analysis_script_sha256': run.sha256_file(__file__),
        'prepared_hash': metadata['prepared_hash'], 'source_hashes': manifest['code_hashes'],
        'selected_epoch_by_validation': selected_epoch,
        'official_selected_test': official['test'] if official is not None else None,
        'test_epoch_range': [expected_epochs[0], expected_epochs[-1]],
        'n_epochs_tested': len(rows), 'lowest_test_error_epoch_post_hoc': ranked[0]['epoch'],
        'lowest_test_error_m_post_hoc': ranked[0]['test']['mean_error_m'],
        'top_10_by_test_error': ranked[:10],
        'ranked_epochs_by_test_error': [row['epoch'] for row in ranked],
        'per_epoch': rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    run.write_json(output, report)
    for rank, row in enumerate(ranked[:10], 1):
        print('EXPLORATORY_TEST_TOP=' + json.dumps({
            'rank': rank, 'epoch': row['epoch'], 'test_mean_error_m': row['test']['mean_error_m'],
            'test_floor_accuracy_pct': row['test']['floor_accuracy_pct'],
            'validation_mean_error_m': row['validation']['mean_error_m']},
            allow_nan=False), flush=True)
    print('EXPLORATORY_TEST_SWEEP_COMPLETE=' + json.dumps({
        'output': str(output), 'n_epochs_tested': len(rows),
        'lowest_test_error_epoch_post_hoc': ranked[0]['epoch'],
        'lowest_test_error_m_post_hoc': ranked[0]['test']['mean_error_m']},
        allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True, help='New JSON file; existing files are never overwritten')
    parser.add_argument('--device', help='Optional CPU/CUDA override; training precision is preserved')
    args = parser.parse_args()
    sweep(args.run_dir, args.output, args.device)


if __name__ == '__main__':
    main()
