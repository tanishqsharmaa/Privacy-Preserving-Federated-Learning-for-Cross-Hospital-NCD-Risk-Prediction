"""
Federated Curriculum Learning for NCD Risk Prediction.

Implements a 3-phase training curriculum that adapts the learning
strategy based on FL round progression:

Phase 1 (Warmup, rounds 1-15):
  - Heavy minority-class oversampling
  - Lower learning rate with linear warm-up
  - Standard BCE loss (simpler than focal for initial warm-up)

Phase 2 (Balanced, rounds 16-35):
  - Focal loss with moderate gamma
  - Natural class ratios (rely on focal loss weighting)
  - Full learning rate

Phase 3 (Hard Mining, rounds 36-50):
  - Focal loss with high gamma (focus on hardest examples)
  - Upweight consistently misclassified minority samples
  - Cosine LR decay

Novel contribution: Curriculum learning in FL for multi-task health
prediction, adapting sampling strategy and loss function to training phase.
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ppfl-ncd.curriculum")


@dataclass
class CurriculumPhaseConfig:
    """Configuration for a single curriculum phase."""
    name: str
    end_round: int  # Phase ends after this round (inclusive)
    loss_type: str  # 'bce', 'focal', 'weighted_bce'
    focal_gamma: float  # Only used for focal loss
    sampling_mode: str  # 'heavy_oversample', 'moderate', 'hard_mine'
    lr_multiplier: float  # Multiplier on base LR
    oversample_factor: float  # How aggressively to oversample (1.0 = normal)


# Default 3-phase curriculum for NCD prediction
DEFAULT_CURRICULUM = [
    CurriculumPhaseConfig(
        name="warmup",
        end_round=15,
        loss_type="weighted_bce",
        focal_gamma=0.0,
        sampling_mode="heavy_oversample",
        lr_multiplier=1.0,  # LR warmup handled by server schedule
        oversample_factor=5.0,  # 5x oversampling of minority
    ),
    CurriculumPhaseConfig(
        name="balanced",
        end_round=35,
        loss_type="focal",
        focal_gamma=2.0,
        sampling_mode="moderate",
        lr_multiplier=1.0,
        oversample_factor=2.0,  # 2x oversampling
    ),
    CurriculumPhaseConfig(
        name="hard_mine",
        end_round=50,
        loss_type="focal",
        focal_gamma=4.0,  # Higher gamma focuses on very hard examples
        sampling_mode="hard_mine",
        lr_multiplier=0.5,  # Reduce LR for fine-tuning
        oversample_factor=1.0,
    ),
]


class CurriculumScheduler:
    """
    Manages the training curriculum across FL rounds.

    The server broadcasts the current curriculum config to clients
    each round, and clients adjust their local training accordingly.
    """

    def __init__(
        self,
        phases: Optional[List[CurriculumPhaseConfig]] = None,
        total_rounds: int = 50,
    ):
        self.phases = phases or DEFAULT_CURRICULUM
        self.total_rounds = total_rounds

        # Validate phases
        if self.phases:
            # Ensure phases cover all rounds
            last_end = self.phases[-1].end_round
            if last_end < total_rounds:
                # Extend last phase
                self.phases[-1].end_round = total_rounds

        logger.info("Curriculum schedule:")
        for phase in self.phases:
            logger.info(
                f"  {phase.name}: rounds 1-{phase.end_round} | "
                f"loss={phase.loss_type} | gamma={phase.focal_gamma} | "
                f"sampling={phase.sampling_mode} | "
                f"oversample={phase.oversample_factor}x"
            )

    def get_phase(self, round_num: int) -> CurriculumPhaseConfig:
        """Get the curriculum phase for a given FL round."""
        for phase in self.phases:
            if round_num <= phase.end_round:
                return phase
        # Default to last phase
        return self.phases[-1]

    def get_round_config(self, round_num: int) -> Dict:
        """
        Get the full configuration for a given round.

        Returns a dict that can be serialized into Flower fit config.
        """
        phase = self.get_phase(round_num)
        return {
            "curriculum_phase": phase.name,
            "curriculum_loss_type": phase.loss_type,
            "curriculum_focal_gamma": phase.focal_gamma,
            "curriculum_sampling_mode": phase.sampling_mode,
            "curriculum_lr_multiplier": phase.lr_multiplier,
            "curriculum_oversample_factor": phase.oversample_factor,
        }

    def get_phase_name(self, round_num: int) -> str:
        """Get the name of the current phase."""
        return self.get_phase(round_num).name

    def get_progress(self, round_num: int) -> Dict:
        """Get human-readable progress info."""
        phase = self.get_phase(round_num)
        prev_end = 0
        for p in self.phases:
            if p.name == phase.name:
                break
            prev_end = p.end_round

        phase_progress = (round_num - prev_end) / (phase.end_round - prev_end)
        overall_progress = round_num / self.total_rounds

        return {
            "phase": phase.name,
            "phase_progress": f"{phase_progress:.0%}",
            "overall_progress": f"{overall_progress:.0%}",
            "round": round_num,
            "total_rounds": self.total_rounds,
        }
