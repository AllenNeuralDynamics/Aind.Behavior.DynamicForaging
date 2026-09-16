import logging
from typing import Literal, TypeVar

from pydantic import Field, SerializeAsAny

from ...trial_models import Trial, TrialMetrics, TrialOutcome
from ...utils import calculate_bias
from .._base import BaseTrialGeneratorSpecModel, ITrialGenerator
from ..composite_trial_generator import TrialGeneratorComposite, TrialGeneratorCompositeSpec
from ..block_based_trial_generator import BlockBasedTrialGenerator

logger = logging.getLogger(__name__)

_TSpec = TypeVar("_TSpec", bound=BaseTrialGeneratorSpecModel, covariant=True)


class CompositeWarmupTrialGeneratorSpec(TrialGeneratorCompositeSpec[_TSpec]):
    """Specification for a composite trial generator that handles warmup to main stage transition.

    This generator ensures that session-level statistics (outcome history, reward history,
    choice history, bias, and session start time) are preserved when transitioning from
    a warmup stage generator to a main stage generator, while resetting trial-specific
    state like current block and baiting state.
    """

    type: Literal["CompositeWarmupTrialGenerator"] = "CompositeWarmupTrialGenerator"

    generators: list[SerializeAsAny[_TSpec]] = Field(
        description="List of trial generator specifications to concatenate. "
        "First generator should be warmup stage, second should be main stage. "
        "When warmup generator returns None, transitions to main stage with state transfer.",
        min_length=2,
        max_length=2,
    )

    def create_generator(self) -> "CompositeWarmupTrialGenerator":
        return CompositeWarmupTrialGenerator(self)


class CompositeWarmupTrialGenerator(TrialGeneratorComposite):
    """Composite trial generator that handles warmup to main stage transitions.

    This generator manages the transition from a warmup stage to a main stage,
    ensuring that session-level statistics persist across the boundary while
    resetting trial-specific state.

    Session statistics that are preserved:
    - outcome_history: Complete record of all trial outcomes
    - reward_history: Complete record of all rewards
    - is_right_choice_history: Complete record of all choices
    - start_time: Original session start time
    - bias: Calculated bias based on cumulative history

    Trial-specific state that is reset:
    - block: Current block (starts fresh in main stage)
    - is_left_baited/is_right_baited: Baiting state (resets at stage boundary)
    - bias_intervention: Reinitialized for main stage
    """

    def __init__(self, spec: CompositeWarmupTrialGeneratorSpec[BaseTrialGeneratorSpecModel]) -> None:
        """Initialize the composite warmup trial generator.

        Args:
            spec: The specification containing warmup and main stage generator specs
        """
        super().__init__(spec)
        if len(self._generators) != 2:
            raise ValueError(
                f"CompositeWarmupTrialGenerator requires exactly 2 generators "
                f"(warmup and main), got {len(self._generators)}"
            )
        logger.debug("Initialized CompositeWarmupTrialGenerator with warmup and main stage generators")

    def next(self) -> Trial | None:
        """Get the next trial, handling state transfer at warmup to main transition.

        If the warmup generator returns None, automatically advances to the main
        generator and transfers session-level state from warmup to main.
        Returns None only when all generators are exhausted.

        Returns:
            The next Trial, or None if all generators are exhausted
        """
        while self._current_index < len(self._generators):
            trial = self._generators[self._current_index].next()

            if trial is not None:
                return trial

            # Current generator returned None, move to next
            if self._current_index == 0:  # Transitioning from warmup (index 0) to main (index 1)
                logger.info("Warmup stage ended, transitioning to main stage with state transfer")
                warmup_generator = self._generators[0]
                self._current_index += 1
                main_generator = self._generators[self._current_index]

                # Transfer session state from warmup to main if both are block-based generators
                if isinstance(warmup_generator, BlockBasedTrialGenerator) and isinstance(
                    main_generator, BlockBasedTrialGenerator
                ):
                    self._transfer_session_state(warmup_generator, main_generator)
                    logger.debug(
                        f"Transferred session state: {len(warmup_generator.outcome_history)} trials from warmup to main"
                    )
            else:
                self._current_index += 1

        # Return None if all generators are exhausted
        return None

    @staticmethod
    def _transfer_session_state(
        warmup_generator: BlockBasedTrialGenerator, main_generator: BlockBasedTrialGenerator
    ) -> None:
        """Transfer session-level state from warmup to main stage generator.

        Args:
            warmup_generator: The warmup stage generator to transfer state from
            main_generator: The main stage generator to transfer state to
        """
        # Transfer cumulative session statistics
        main_generator.outcome_history = warmup_generator.outcome_history.copy()
        main_generator.is_right_choice_history = warmup_generator.is_right_choice_history.copy()
        main_generator.reward_history = warmup_generator.reward_history.copy()
        main_generator.start_time = warmup_generator.start_time

        # Recalculate bias with transferred history
        main_generator.bias = calculate_bias(
            outcomes=main_generator.outcome_history,
            outcome_window_length=(
                200
                if not main_generator.spec.bias_intervention_parameters
                else main_generator.spec.bias_intervention_parameters.bias_window_length
            ),
        )

        # Reset trial-specific state (baiting state is reset at stage boundary)
        main_generator.is_left_baited = False
        main_generator.is_right_baited = False

        logger.debug(
            f"Session state transferred: {len(main_generator.outcome_history)} outcomes, "
            f"{sum(main_generator.reward_history)} rewards, "
            f"bias={main_generator.bias:.3f}"
        )
