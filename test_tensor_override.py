"""Regression checks for generated commands and Gemma shared-KV placement."""
import contextlib
import importlib.util
import io
import pathlib
import shlex
import sys
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    'tensor_override', pathlib.Path(__file__).with_name('tensor-override.py'))
planner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = planner
spec.loader.exec_module(planner)


def model_fixture(arch='gemma4', layers=6):
    md = {
        'general.architecture': arch,
        f'{arch}.block_count': layers,
        f'{arch}.context_length': 8192,
        f'{arch}.embedding_length': 32,
        f'{arch}.attention.head_count': 4,
        f'{arch}.attention.head_count_kv': 2,
        f'{arch}.attention.sliding_window': 1024,
        f'{arch}.attention.sliding_window_pattern': [True] * (layers - 1) + [False],
    }
    tensors = [planner.Tensor('token_embd.weight', (32, 256), 1)]
    for il in range(layers):
        for name in ('attn_q', 'attn_k', 'attn_v', 'ffn_up'):
            tensors.append(planner.Tensor(f'blk.{il}.{name}.weight', (32, 64), 1))
    return md, tensors


class PlannerRegressionTests(unittest.TestCase):
    def setUp(self):
        planner.MTP_SIDECAR = None
        planner.MTP_DRAFT_DEV = 'CUDA0'
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = pathlib.Path(self.tmp.name, 'target.gguf')
        self.sidecar = pathlib.Path(self.tmp.name, 'draft with spaces.gguf')
        self.model.touch()
        self.sidecar.touch()
        self.target_data = model_fixture()
        self.draft_data = model_fixture('gemma4-assistant', 4)
        self.gpus = [planner.Gpu(i, f'GPU{i}', mib * planner.MiB,
                                mib * planner.MiB, mib * planner.MiB)
                     for i, mib in enumerate((16000, 8000))]

    def run_planner(self, *options):
        def read(path):
            return self.draft_data if str(path) == str(self.sidecar) else self.target_data
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, 'argv', ['tensor-override.py', str(self.model),
                                            '--no-tensor-split', '-np', '1', *options]), \
                mock.patch.object(planner, 'read_gguf', side_effect=read), \
                mock.patch.object(planner, 'detect_gpus', return_value=self.gpus), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            planner.main()
        return shlex.split(out.getvalue().replace('\\\n', ' ')), err.getvalue()

    def test_explicit_dense_context_is_preserved(self):
        command, _ = self.run_planner('-q8', '-c', '2048', '--no-mtp')
        self.assertEqual(command[command.index('-c') + 1], '2048')

    def test_infeasible_context_does_not_emit_a_smaller_context(self):
        with self.assertRaisesRegex(SystemExit, 'requested -c 1000000000 does not fit'):
            self.run_planner('-q8', '-c', '1000000000', '--no-mtp')

    def test_cuda0_mtp_keeps_both_cache_backends_visible(self):
        command, _ = self.run_planner('-q8', '-c', '2048', '--mtp-model', str(self.sidecar))
        self.assertEqual(command[command.index('--spec-draft-device') + 1], 'CUDA0,CUDA1')
        self.assertEqual(command[command.index('--spec-draft-override-tensor') + 1], '.*=CUDA0')
        self.assertEqual(command[command.index('--spec-draft-model') + 1], str(self.sidecar))

    def test_cuda1_remains_selectable(self):
        command, _ = self.run_planner('-q8', '-c', '2048', '--mtp-model', str(self.sidecar),
                                      '--mtp-device', 'CUDA1')
        self.assertEqual(command[command.index('--spec-draft-override-tensor') + 1], '.*=CUDA1')

    def test_memory_relevant_options_reach_the_command(self):
        command, _ = self.run_planner('-q8', '-c', '2048', '--mtp-model', str(self.sidecar),
                                      '-b', '1024', '--swa-full', '--draft-max', '5')
        self.assertEqual(command[command.index('-b') + 1], '1024')
        self.assertIn('--swa-full', command)
        self.assertEqual(command[command.index('--spec-draft-n-max') + 1], '5')

    def test_sidecar_embedding_is_not_double_charged(self):
        with mock.patch.object(planner, 'read_gguf', return_value=self.draft_data):
            sidecar = planner.load_mtp_sidecar(str(self.sidecar))
        self.assertEqual(sidecar['weights'], sum(t.nbytes for t in self.draft_data[1]))
        self.assertEqual(sidecar['cpu_weights'], 0)

    def test_gemma_scratch_covers_measured_context_growth(self):
        info = planner.analyse(*self.target_data)
        for ctx, observed_mib in ((32768, 522.20), (71680, 978.20)):
            estimated = planner.dense_compute_buffer(info, ctx, 512, 1) / planner.MiB
            self.assertGreaterEqual(estimated, observed_mib)
            self.assertLess(estimated - observed_mib, 2)

    def test_dense_measure_is_used_at_requested_context(self):
        measured = ({'CUDA0': 600 * planner.MiB, 'CUDA1': 600 * planner.MiB},
                    {'CUDA0': 450 * planner.MiB})
        with mock.patch.object(planner, 'measure_compute_buffers', return_value=measured) as probe:
            command, report = self.run_planner('-q8', '-c', '2048', '--measure',
                                               '--mtp-model', str(self.sidecar))
        self.assertTrue(probe.called)
        probe_command = shlex.split(probe.call_args.kwargs['command'].replace('\\\n', ' '))
        self.assertEqual(probe_command[probe_command.index('-c') + 1], '2048')
        self.assertEqual(command[command.index('-c') + 1], '2048')
        self.assertIn('compute buffer (measured)', report)

    def test_measure_parses_external_draft_and_multiline_command(self):
        process = mock.Mock()
        process.stdout = io.StringIO(
            'sched_reserve: CUDA0 compute buffer size = 500.00 MiB\n'
            'sched_reserve: CUDA1 compute buffer size = 500.00 MiB\n'
            'sched_reserve: graph nodes = 300\n'
            'common_speculative_init_result: loading draft model\n'
            'sched_reserve: CUDA0 compute buffer size = 400.00 MiB\n'
            'sched_reserve: graph nodes = 150\n'
            'llama_server: listening on http://127.0.0.1:18099\n')
        with mock.patch.object(planner.subprocess, 'Popen', return_value=process) as spawn:
            command, report = self.run_planner('-q8', '-c', '2048', '--measure',
                                               '--llama-server', sys.executable,
                                               '--mtp-model', str(self.sidecar))
        spawned_command = spawn.call_args.args[0]
        self.assertFalse(any('\n' in arg for arg in spawned_command))
        self.assertEqual(spawned_command[spawned_command.index('--spec-draft-override-tensor') + 1],
                         '.*=CUDA0')
        self.assertIn('MTP ctx: CUDA0', report)
        process.terminate.assert_called_once()

    def test_failed_measurement_does_not_accept_dry_run_buffers(self):
        process = mock.Mock()
        process.stdout = io.StringIO(
            'sched_reserve: CUDA0 compute buffer size = 500.00 MiB\n'
            'sched_reserve: graph nodes = 300\n'
            'cudaMalloc failed: out of memory\n')
        with mock.patch.object(planner.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(SystemExit, 'did not reach readiness'):
                self.run_planner('-q8', '-c', '2048', '--measure',
                                 '--llama-server', sys.executable)

    def test_cross_gpu_shared_cache_adds_transfer_scratch(self):
        info = planner.analyse(*self.target_data)
        args = type('Args', (), dict(parallel=1, ubatch=512, kv_unified=True, swa_full=False))()
        planner.MTP_SIDECAR = {'arch': 'gemma4-assistant'}
        local = planner.sidecar_transfer_bytes(info, args, 32768, 8, lambda il: 0)
        remote = planner.sidecar_transfer_bytes(info, args, 32768, 8, lambda il: 1)
        self.assertEqual(local, 0)
        self.assertGreater(remote, 0)

    def test_shared_kv_draft_does_not_get_an_incompatible_tensor_plan(self):
        planner.MTP_SIDECAR = {'shares_kv': True}
        self.assertIsNone(planner.tensor_fit(None, None, None, None, None, 0, 32768,
                                             8, None, True))


if __name__ == '__main__':
    unittest.main()
