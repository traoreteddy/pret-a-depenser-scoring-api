"""Politique d'échantillonnage des entrées journalisées (maîtrise du coût de stockage).

- 100 % des prédictions reçoivent une ligne légère (score, décision, latences) ;
- les entrées complètes (raw_input + features, ~2 Ko) ne sont conservées que si :
    * tirage déterministe sur le request_id < taux (`LOG_SAMPLE_RATE`), ou
    * score en zone grise autour du seuil (`LOG_GREY_ZONE`) — les cas utiles au calibrage, ou
    * trafic explicitement marqué (en-tête X-Scenario) — mesure de la dérive simulée.
"""

from __future__ import annotations

import hashlib


def deterministic_unit(request_id: str) -> float:
    """Nombre dans [0, 1) dérivé du request_id : même requête → même tirage (rejouable)."""
    digest = hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def should_sample(
    request_id: str,
    proba: float | None,
    *,
    threshold: float,
    rate: float,
    grey_zone: float,
    forced: bool = False,
) -> bool:
    if forced:
        return True
    if proba is not None and abs(proba - threshold) < grey_zone:
        return True
    return deterministic_unit(request_id) < rate
