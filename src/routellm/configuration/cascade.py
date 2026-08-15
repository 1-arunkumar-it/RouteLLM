"""Validated cascade router configuration.

Configuration is the only place where cascade knobs are defined. It must
never contain secrets and every value is validated at construction time.
"""

from dataclasses import dataclass

_CALIBRATION_METHODS = ("sigmoid", "isotonic")


@dataclass(frozen=True)
class CascadeConfig:
    """Settings for the cascaded routing engine.

    ``rule_override_min_precision`` is the minimum held-out rule precision a
    category needs before its rule decision may bypass the classifier
    (SPEC section 48). ``threshold`` optionally overrides a persisted model's
    validation-selected threshold for one ``RouteService`` instance; when
    ``None``, the model's validated threshold is used. Cascade training always
    persists the threshold selected from validation data, and the CLI exposes
    no runtime override. ``calibration_cv`` and ``calibration_method`` control
    how classifier probabilities are calibrated (SPEC section 20).
    """

    rule_override_min_precision: float = 0.90
    threshold: float | None = None
    calibration_cv: int = 5
    calibration_method: str = "sigmoid"

    def __post_init__(self) -> None:
        if not 0 <= self.rule_override_min_precision <= 1:
            raise ValueError(
                "rule_override_min_precision must be in [0, 1], got "
                f"{self.rule_override_min_precision}."
            )
        if self.threshold is not None and not 0 <= self.threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1] or None, got {self.threshold}.")
        if self.calibration_cv < 2:
            raise ValueError(f"calibration_cv must be >= 2, got {self.calibration_cv}.")
        if self.calibration_method not in _CALIBRATION_METHODS:
            raise ValueError(
                f"calibration_method must be one of {_CALIBRATION_METHODS}, "
                f"got {self.calibration_method!r}."
            )
