"""Backend « xgb_native » : Booster XGBoost chargé directement (sans mlflow, sklearn ni pandas).

`inplace_predict` évite la construction d'une DMatrix et d'un DataFrame ; `nthread=1` évite la
contention entre requêtes concurrentes (l'API parallélise déjà au niveau des requêtes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from scoring_api.features.spec import FEATURES
from scoring_api.model.base import ModelInfo
from scoring_api.model.meta import read_meta


class XgbNativePredictor:
    def __init__(
        self, model_dir: Path, threshold_override: float | None = None, nthread: int = 1
    ) -> None:
        path = model_dir / "booster.ubj"
        if not path.exists():
            msg = f"{path} absent : exécuter scripts/export_model.py"
            raise FileNotFoundError(msg)
        meta = read_meta(model_dir)
        if "threshold" not in meta:
            msg = "model_meta.json sans seuil : exécuter scripts/import_model.py"
            raise ValueError(msg)
        self._booster = xgb.Booster()
        self._booster.load_model(str(path))
        self._booster.set_param({"nthread": nthread})
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
            backend="xgb_native",
            threshold=threshold_override
            if threshold_override is not None
            else float(meta["threshold"]),
            run_id=meta.get("run_id"),
        )

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        out = self._booster.inplace_predict(
            np.ascontiguousarray(x[:, self._perm], dtype=np.float32), validate_features=False
        )
        return np.asarray(out, dtype=np.float64).reshape(-1)
