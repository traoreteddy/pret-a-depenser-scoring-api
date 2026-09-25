"""Exporte le modèle champion sous des formats d'inférence rapides.

- booster.ubj : le booster XGBoost natif (format binaire universel, stable entre versions) ;
- model.onnx : conversion ONNX (onnxmltools) pour ONNX Runtime ;
- model_meta.json : complété (feature_names, exports, sha256).

Prérequis : models/scoring_credit_v1 importé (scripts/import_model.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/scoring_credit_v1"))
    parser.add_argument("--no-onnx", action="store_true")
    args = parser.parse_args()
    model_dir: Path = args.model_dir

    from scoring_api.features.spec import FEATURES  # noqa: PLC0415
    from scoring_api.model.pyfunc_backend import PyfuncPredictor  # noqa: PLC0415

    pyfunc = PyfuncPredictor(model_dir)
    # ScoringCredit.modele = FixedThresholdClassifier(FrozenEstimator(Pipeline(prep, model)))
    pipeline = pyfunc._inner.modele.estimator.estimator
    xgb_clf = pipeline.named_steps["model"]
    booster = xgb_clf.get_booster()
    # Le ColumnTransformer du pipeline préfixe et réordonne les colonnes (ex. "num__AMT_ANNUITY") :
    # on enregistre la permutation spec (ordre de la signature) → ordre attendu par le booster.
    raw_names = list(booster.feature_names or FEATURES)
    names = [n.split("__", 1)[-1] for n in raw_names]
    if sorted(names) != sorted(FEATURES):
        msg = f"Features du booster ≠ spec : {sorted(set(names) ^ set(FEATURES))}"
        raise ValueError(msg)
    perm = [FEATURES.index(n) for n in names]

    booster_path = model_dir / "booster.ubj"
    booster.save_model(str(booster_path))
    print(
        f"booster.ubj écrit ({booster_path.stat().st_size / 1e3:.0f} Ko, {booster.num_boosted_rounds()} arbres)"
    )

    with (model_dir / "input_example.json").open(encoding="utf-8") as f:
        ex = json.load(f)
    x = np.array(ex["data"], dtype=np.float64)[:, [ex["columns"].index(c) for c in FEATURES]]
    ref = pyfunc.predict_proba(x)

    import xgboost as xgb  # noqa: PLC0415

    b2 = xgb.Booster()
    b2.load_model(str(booster_path))
    native = b2.inplace_predict(x[:, perm].astype(np.float32), validate_features=False)
    print(f"écart max booster natif vs pyfunc : {np.abs(native - ref).max():.2e}")

    exports: dict[str, object] = {
        "booster_ubj": {"sha256": sha256(booster_path), "xgboost_version": xgb.__version__}
    }
    if not args.no_onnx:
        import onnx  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415
        from onnxmltools.convert import convert_xgboost  # noqa: PLC0415
        from onnxmltools.convert.common.data_types import FloatTensorType  # noqa: PLC0415

        # Le convertisseur exige des noms f0..f37 : on convertit un booster rechargé sans noms.
        plain = xgb.Booster()
        plain.load_model(str(booster_path))
        plain.feature_names = None
        plain.feature_types = None
        onnx_model = convert_xgboost(
            plain,
            initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
            target_opset=15,
        )
        onnx_path = model_dir / "model.onnx"
        onnx.save_model(onnx_model, str(onnx_path))
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        outputs = sess.run(None, {"input": x[:, perm].astype(np.float32)})
        proba_onnx = (
            np.array([p[1] for p in outputs[1]])
            if isinstance(outputs[1][0], dict)
            else np.asarray(outputs[1])[:, 1]
        )
        print(
            f"model.onnx écrit ({onnx_path.stat().st_size / 1e3:.0f} Ko) ; écart max ONNX vs pyfunc : {np.abs(proba_onnx - ref).max():.2e}"
        )
        exports["onnx"] = {
            "sha256": sha256(onnx_path),
            "onnxruntime_version": ort.__version__,
            "opset": 15,
        }

    meta_path = model_dir / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.pop("feature_names", None)
    meta.update(
        {
            "booster_feature_names": list(names),
            "booster_feature_order": perm,
            "exports": exports,
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("model_meta.json mis à jour")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
