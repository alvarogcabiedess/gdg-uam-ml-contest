"""One-thread CPU candidate built only from our frozen weights."""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from preprocessing import preprocess_into, warmup

HERE = Path(__file__).resolve().parent
ort.disable_telemetry_events()


class Model:
    def __init__(self):
        self.config = json.loads((HERE/'runtime.json').read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(HERE/'model.onnx'), options, providers=['CPUExecutionProvider'])
        self.series = np.zeros((1, 4, 256), np.float32)
        self.binding = self.session.io_binding()
        self.binding.bind_cpu_input('series', self.series.reshape(1, 4, 1, 256))
        self.meta = np.zeros((1, 1), np.float32)
        self.logits = np.empty((1, 8), np.float32)
        self.hybrid = self.config['hybrid']
        self.scratch = self.config['scratch']
        if self.scratch:
            self.scratch_arrays = (np.empty((4, 256), np.float32), np.empty(16, np.int64), np.empty((2, 16), np.float64))
            raw = np.zeros((4, 256), np.float32)
            preprocess_into(raw, self.series, *self.scratch_arrays)
            raw.setflags(write=False)
            preprocess_into(raw, self.series, *self.scratch_arrays)
        else:
            warmup()
        if self.hybrid:
            from fused_head import finish
            self.finish = finish
            self.z = np.zeros((1, self.config['channels'], 1, 64), np.float32)
            self.z2 = self.z.reshape(self.config['channels'], 64)
            self.binding.bind_output('activations', 'cpu', 0, np.float32, self.z.shape, self.z.ctypes.data)
            with np.load(HERE/'head.npz') as weights:
                self.head = tuple(weights[k].copy() for k in ('w1', 'b1', 'w2', 'b2'))
            self.features = np.empty(self.head[0].shape[0], np.float32)
            self.hidden = np.empty(self.head[0].shape[1], np.float32)
            self.out = self.logits[0]
            self.finish(self.z2, np.float32(0), *self.head, self.features, self.hidden, self.out)
        else:
            self.binding.bind_cpu_input('meta', self.meta)
            self.binding.bind_output('logits', 'cpu', 0, np.float32, self.logits.shape, self.logits.ctypes.data)

    def predict(self, sample) -> int:
        raw = np.ascontiguousarray(sample.series, dtype=np.float32)
        if self.scratch:
            preprocess_into(raw, self.series, *self.scratch_arrays)
        else:
            preprocess_into(raw, self.series)
        rate = float(sample.meta.get('sample_rate', 1.0))-1.5
        if self.hybrid:
            self.session.run_with_iobinding(self.binding)
            return int(self.finish(self.z2, np.float32(rate), *self.head, self.features, self.hidden, self.out))
        self.meta[0, 0] = rate
        self.session.run_with_iobinding(self.binding)
        return int(self.logits.argmax())
