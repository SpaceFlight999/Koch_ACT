import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from koch_autonomy import replay_auto
from koch_autonomy.planner import plan_task
from koch_project_paths import ROOT, resolve_config_reference


class AutoReplayTests(unittest.TestCase):
    def test_direct_counts_aggregate_demand_for_repeated_colors(self):
        stock = {'S1': ['red'], 'S2': [], 'S3': [], 'S4': []}
        targets = {f'T{i}': {'color': 'red', 'number': int(i < 3)} for i in range(1, 5)}
        with self.assertRaisesRegex(ValueError, "库存不足.*'red': 1"):
            plan_task(stock, targets, quantity_rule='direct')
        stock['S2'] = ['red']
        self.assertEqual(len(plan_task(stock, targets, quantity_rule='direct')['moves']), 2)
        with self.assertRaisesRegex(ValueError, '颜色各异'):
            plan_task(stock, targets)

    def test_chain_selects_compatible_episode_instead_of_latest_incompatible_one(self):
        def episode(path, start, end):
            goals = [{name: ticks for name in replay_auto.NAMES} for ticks in (start, end)]
            return SimpleNamespace(path=Path(path), rows=[{'goal_raw': row} for row in goals])

        stock = {'S1': ['red'], 'S2': ['red'], 'S3': [], 'S4': []}
        targets = {f'T{i}': {'color': 'red', 'number': int(i < 3)} for i in range(1, 5)}
        first = episode('first', 1000, 1000)
        compatible = episode('old', 1050, 1100)
        incompatible = episode('new', 1200, 1100)
        with patch.object(replay_auto, 'route_episodes', side_effect=[[first], [compatible, incompatible]]), \
                patch.object(replay_auto, 'read_trajectory', side_effect=lambda item, *_: item):
            _, selected = replay_auto.prepare_chain(ROOT / 'datasets',
                {'inventory': stock, 'targets': targets}, 'direct', None)
        self.assertEqual(selected, (first, compatible))
        with patch.object(replay_auto, 'route_episodes', side_effect=[[first], [incompatible]]), \
                patch.object(replay_auto, 'read_trajectory', side_effect=lambda item, *_: item):
            with self.assertRaisesRegex(ValueError, '第 1 步接第 2 步.*未连接从臂'):
                replay_auto.prepare_chain(ROOT / 'datasets', {'inventory': stock, 'targets': targets}, 'direct', None)

    def test_preview_reads_site_relative_paths_without_connecting(self):
        with tempfile.TemporaryDirectory(prefix='.replay-site-test-', dir=ROOT) as tmp:
            folder = Path(tmp)
            site = folder / 'replay.json'
            site.write_text(json.dumps({'source': {'config': 'field/source.json', 'camera': 2},
                'targets': {'config': 'field/targets.json', 'rotation': 0, 'camera': 1},
                'quantity_rule': 'direct', 'data': 'my-data'}), encoding='utf-8')
            views = []

            def observe(args):
                views.append(args)
                report = dict(complete=True, stable=True)
                if args.kind == 'source':
                    return dict(report, empty_reference_used=True, inventory={
                        'S1': ['red'], 'S2': [], 'S3': [], 'S4': []})
                return dict(report, targets={f'T{i}': {'color': 'red', 'number': int(i == 1)} for i in range(1, 5)})

            with patch.object(replay_auto, 'ROOT', folder), patch.object(replay_auto, 'connect_bus') as connect, \
                    patch.object(replay_auto, 'prepare_chain',
                        return_value=({'moves': [{'route': 'S1_T1'}]}, [SimpleNamespace(path='demo')])) as chain:
                replay_auto.run_auto(site, True, observe, cancelled=lambda: False)
                connect.assert_not_called()
                self.assertEqual(chain.call_args.args[0], folder / 'my-data')
            self.assertEqual([args.config for args in views],
                             [folder / 'field/source.json', folder / 'field/targets.json'])
            self.assertEqual(resolve_config_reference('../calibration.json', views[0].config),
                             folder / 'calibration.json')


if __name__ == '__main__':
    unittest.main()
