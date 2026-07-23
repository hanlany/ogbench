"""Batch wrapper and summary writer for matched planner runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .evaluation_protocol import classify_failure, summarize_offline_rows
from .plan_antmaze import make_parser, run


def result_row(result: dict[str, Any], task_id: int, environment_seed: int, planner_seed: int) -> dict[str, Any]:
    """Flatten stable scalar fields for an evaluation table."""
    return {
        'task_id': int(task_id),
        'environment_seed': int(environment_seed),
        'planner_seed': int(planner_seed),
        'status': result.get('status'),
        'reason': result.get('reason'),
        'goal_node': result.get('goal_node'),
        'candidate_plan': result.get('goal_node') is not None,
        'planning_time': result.get('planning_time'),
        'iterations': result.get('tree_summary', {}).get('iterations'),
        'first_goal_iteration': result.get('tree_summary', {}).get('first_goal_iteration'),
        'model_propagation_calls': result.get('tree_summary', {}).get('model_propagation_calls'),
        'nodes': result.get('tree_summary', {}).get('nodes'),
        'nodes_to_candidate': result.get('tree_summary', {}).get('nodes')
        if result.get('goal_node') is not None
        else None,
        'predicted_plan_steps': result.get('predicted_plan_steps'),
        'predicted_action_energy': result.get('predicted_action_energy'),
        'minimum_wall_clearance': result.get('minimum_wall_clearance'),
        'minimum_cycle_error_margin': result.get('minimum_cycle_error_margin'),
        'best_predicted_goal_distance': result.get('best_predicted_goal_distance'),
        'execution_success': result.get('execution_success'),
        'minimum_actual_goal_distance': result.get('minimum_actual_goal_distance'),
        'final_actual_goal_distance': result.get('final_actual_goal_distance'),
        'control_proposal_label': result.get('control_proposal_label'),
        'rejection_counters': result.get('rejection_counters', {}),
        'failure_taxonomy': classify_failure(result),
    }


def write_summary(rows: Iterable[dict[str, Any]], output_dir: str | Path) -> None:
    """Write JSON, CSV, and a compact Markdown summary without hiding failures."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    offline_summary = summarize_offline_rows(rows) if rows else {'runs': 0}
    (output_dir / 'metrics.json').write_text(
        json.dumps({'summary': offline_summary, 'rows': rows}, indent=2, sort_keys=True) + '\n'
    )
    fields = list(rows[0]) if rows else ['status']
    with (output_dir / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    statuses = {}
    for row in rows:
        statuses[row.get('status')] = statuses.get(row.get('status'), 0) + 1
    executed = [row for row in rows if row.get('execution_success') is not None]
    success_rate = float(np.mean([row.get('execution_success') is True for row in executed])) if executed else None
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    if rows:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        status_labels = sorted(str(row.get('status')) for row in rows)
        status_labels = sorted(set(status_labels))
        counts = [statuses.get(status, 0) for status in status_labels]
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.bar(status_labels, counts)
        axis.set(title='Offline planner statuses', ylabel='runs')
        axis.tick_params(axis='x', rotation=30)
        figure.savefig(plots_dir / 'status_counts.png', dpi=160)
        plt.close(figure)
    lines = [
        '# Planner evaluation summary',
        '',
        f'Runs: {len(rows)}',
        f'Status counts: {statuses}',
        f'Offline candidate-plan rate: {offline_summary.get("candidate_plan_rate", 0.0):.6f}',
        f'MuJoCo success rate among executed rows: {success_rate if success_rate is not None else "not measured"}',
        '',
        'Predicted success is not simulator success; rows without execution retain a null execution field.',
    ]
    (output_dir / 'summary.md').write_text('\n'.join(lines) + '\n')


def _parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(',') if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError('Expected at least one integer.')
    return values


def main() -> None:
    base_parser = make_parser()
    for action in base_parser._actions:
        if action.dest in ('task_id', 'environment_seed', 'planner_seed', 'output_dir'):
            action.required = False
            if action.dest == 'output_dir':
                action.default = None
    parser = argparse.ArgumentParser(description=__doc__, parents=[base_parser], add_help=False)
    parser.add_argument('--task_ids', default='1,2,3,4,5', type=_parse_ints)
    parser.add_argument('--environment_seeds', default='0', type=_parse_ints)
    parser.add_argument('--planner_seeds', default='0', type=_parse_ints)
    parser.add_argument('--evaluation_output_dir', required=True)
    args = parser.parse_args()
    root = Path(args.evaluation_output_dir).expanduser().resolve()
    rows = []
    for task_id in args.task_ids:
        for environment_seed in args.environment_seeds:
            for planner_seed in args.planner_seeds:
                run_args = argparse.Namespace(**vars(args))
                run_args.task_id = task_id
                run_args.environment_seed = environment_seed
                run_args.planner_seed = planner_seed
                run_args.output_dir = str(root / f'task{task_id}_env{environment_seed}_plan{planner_seed}')
                result = run(run_args)
                rows.append(result_row(result, task_id, environment_seed, planner_seed))
    write_summary(rows, root)


if __name__ == '__main__':
    main()
