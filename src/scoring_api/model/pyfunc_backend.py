"""Backend « pyfunc » : chargement du modèle MLflow tel qu'exporté au P6 (baseline v1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scoring_api.features.spec import FEATURES, INT_FEATURES
from scoring_api.model.base import ModelInfo
from scoring_api.model.meta import read_meta, read_mlmodel, signature_inputs


class PyfuncPredictor:
    """Encapsule `mlflow.pyfunc.PyFuncModel`. Les dépendances lourdes sont importées ici seulement."""

    def __init__(self, model_dir: Path, threshold_override: float | None = None) -> None:
        import mlflow.pyfunc  # noqa: PLC0415 - import différé (dépendance optionnelle)
        import pandas as pd  # noqa: PLC0415

        self._pd = pd
        self._model = mlflow.pyfunc.load_model(str(model_dir))
        self._inner: Any = self._model.unwrap_python_model()
        sig = signature_inputs(model_dir)
        if tuple(sig) != FEATURES:
            msg = f"Signature MLflow ≠ spec des features : {sig[:5]}…"
            raise ValueError(msg)
        self.features = FEATURES
        mlmodel = read_mlmodel(model_dir)
        meta = read_meta(model_dir)
        threshold = float(self._inner.seuil)
        self.info = ModelInfo(
            name=meta.get("name", "scoring_credit"),
            version=str(meta.get("version", "1")),
            backend="pyfunc",
            threshold=threshold_override if threshold_override is not None else threshold,
            run_id=mlmodel.get("run_id"),
        )
        self.native_threshold = threshold
        self._dtypes = {f: ("int64" if f in INT_FEATURES else "float64") for f in FEATURES}

    def _frame(self, x: NDArray[np.float64]) -> Any:
        df = self._pd.DataFrame(x, columns=list(FEATURES))
        return df.astype(self._dtypes, copy=False)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # On appelle directement le classifieur sous-jacent : évite la validation de schéma
        # MLflow (coûteuse) tout en restant identique au pyfunc (mêmes poids, même pipeline).
        proba = self._inner.modele.predict_proba(self._frame(x))[:, 1]
        return np.asarray(proba, dtype=np.float64)

    def predict_pyfunc(self, x: NDArray[np.float64]) -> Any:
        """Chemin MLflow complet (validation de signature incluse) — utilisé pour les tests."""
        return self._model.predict(self._frame(x))
