"""Backend « onnx » : ONNX Runtime (CPU), session unique, un thread intra-op."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scoring_api.features.spec import FEATURES
from scoring_api.model.base import ModelInfo
from scoring_api.model.meta import read_meta


class OnnxPredictor:
    def __init__(
        self, model_dir: Path, threshold_override: float | None = None, intra_threads: int = 1
    ) -> None:
        import onnxruntime as ort  # noqa: PLC0415 - dépendance optionnelle

        path = model_dir / "model.onnx"
        if not path.exists():
            msg = f"{path} absent : exécuter scripts/export_model.py"
            raise FileNotFoundError(msg)
        meta = read_meta(model_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name
        self._proba_output = self._sess.get_outputs()[1].name
        perm = meta.get("booster_feature_order")
        if not perm or len(perm) != len(FEATURES):
            msg = (
                "booster_feature_order absent de model_meta.json : exécuter scripts/export_model.py"
            )
            raise ValueError(msg)
        self._perm = np.asarray(perm, dtype=np.intp)
        self.features = FEATURES
        self.info = ModelInfo(
            name=str(meta.get("name", "scoring_credit")),
            version=str(meta.get("version", "1")),
            backend="onnx",
            threshold=threshold_override
            if threshold_override is not None
            else float(meta["threshold"]),
            run_id=meta.get("run_id"),
        )

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        out = self._sess.run(
            [self._proba_output],
            {self._input: np.ascontiguousarray(x[:, self._perm], dtype=np.float32)},
        )[0]
        if isinstance(out, list):  # ZipMap : liste de dicts {classe: proba}
            return np.array([float(d[1]) for d in out], dtype=np.float64)
        return np.asarray(out, dtype=np.float64)[:, 1]
