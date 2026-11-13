# Scopo: contratti minimi per sorgenti embedding, campioni spatial/tabular e protocollo MSA.
# Fasi: validazione shape/metadati -> configurazione modello/protocollo separata -> errori prima del fit.
# Input: array locali e YAML risolto; output: dataclass/protocol strutturali, nessuna acquisizione.
# Parametri: C configurabile, kernel 1/3, proiezione opzionale, normalizzazione dichiarata.
# Esempio: ModelConfig(input_channels=384, projection_channels=64). Risorse: CPU minima.
# Ripresa: identità feature-space e configurazione sono parte del checkpoint; nessun riuso implicito.
from dataclasses import asdict, dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = 'spatial_cnn'
    input_channels: int = 128
    projection_channels: int | None = None
    kernel: int = 3
    conv_channels: tuple = (64, 32)
    hidden: int = 64
    dropout: float = 0.1
    output: str = 'linear'

    def __post_init__(self):
        if self.architecture not in {'spatial_cnn', 'mean_mlp'} or self.kernel not in {1, 3}:
            raise ValueError('Architettura/kernel non implementati')
        if self.input_channels < 1 or self.hidden < 1 or len(self.conv_channels) != 2 or min(self.conv_channels) < 1:
            raise ValueError('Dimensioni modello non valide')
        if self.projection_channels is not None and (self.projection_channels < 1 or self.architecture != 'spatial_cnn'):
            raise ValueError('Proiezione valida soltanto per CNN, con D positivo')
        if not 0 <= self.dropout < 1 or self.output not in {'linear', 'sigmoid'}:
            raise ValueError('Dropout/output non valido')


@dataclass(frozen=True)
class ExperimentProtocol:
    dataset: str = 'biomap_globio4'
    target: str = 'globio4_msa_2020'
    year: int = 2020
    profile: str = 'benchmark'
    normalization: str = 'pixel_weighted_standard'
    support: str = 'preserve_legacy_contributions'
    seed: int = 42
    selection: str = 'oof_macro_aoi_rmse'

    def __post_init__(self):
        if self.profile not in {'benchmark', 'synthetic'}:
            raise ValueError('Profilo sconosciuto')
        if self.year != 2020 or self.target != 'globio4_msa_2020':
            raise ValueError('Questo profilo richiede GLOBIO4 MSA 2020')
        if self.normalization not in {'pixel_weighted_standard', 'cell_standard', 'none'}:
            raise ValueError('Normalizzazione sconosciuta')
        if self.support != 'preserve_legacy_contributions' or self.selection != 'oof_macro_aoi_rmse':
            raise ValueError('Supporto/selezione non implementati: nessun fallback')


@dataclass
class SpatialSample:
    features: np.ndarray
    mask: np.ndarray
    metadata: dict

    def __post_init__(self):
        if self.features.ndim != 3 or self.mask.shape != self.features.shape[1:] or self.mask.dtype != np.bool_:
            raise ValueError('SpatialSample richiede [C,H,W] e mask bool [H,W]')
        if not self.mask.any() or not np.isfinite(self.features[:, self.mask]).all():
            raise ValueError('Supporto vuoto o embedding validi non finiti')
        if not self.metadata.get('feature_space_id'):
            raise ValueError('feature_space_id obbligatorio')


@dataclass
class TabularSample:
    features: np.ndarray
    metadata: dict

    def __post_init__(self):
        if self.features.ndim != 1 or not np.isfinite(self.features).all():
            raise ValueError('TabularSample richiede un vettore finito, non una mappa')


class EmbeddingSource(Protocol):
    channels: int
    feature_space_id: str
    fingerprint: str

    def patches(self, cell_id: str) -> list[SpatialSample]: ...


class Predictor(Protocol):
    def __call__(self, batch): ...


@dataclass(frozen=True)
class ExperimentSpec:
    source: dict
    model: ModelConfig
    protocol: ExperimentProtocol
    training: dict
    execution: dict

    def __post_init__(self):
        if not self.source.get('feature_space_id'):
            raise ValueError('Specificare feature_space_id; encoder diversi non sono intercambiabili')
        if self.model.architecture == 'spatial_cnn' and self.protocol.normalization == 'cell_standard':
            raise ValueError('cell_standard e definita sui vettori: usare mean_mlp, non mappe')
        if self.source.get('representation', 'spatial') != 'spatial':
            raise ValueError('Questo runner richiede mappe autentiche; V2 Mean rimane tabellare legacy')

    def to_dict(self):
        return asdict(self)
