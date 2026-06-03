from abc import ABC, abstractmethod


class BaseIntervention(ABC):
    @abstractmethod
    def are_intervention_conditions_met() -> bool:
        """Abstract method to determine if intervention conditions are met.

        Returns:
            True if intervention conditions are met, False otherwise.
        """

        pass

    @abstractmethod
    def determine_intervention():
        """Abstract method to determine interventions if conditions are met."""

        pass
