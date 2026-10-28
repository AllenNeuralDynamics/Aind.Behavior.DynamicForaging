from pathlib import Path

from aind_behavior_dynamic_foraging_curricula.coupled_baiting import TRAINER
from aind_behavior_dynamic_foraging_curricula.metrics import DynamicForagingMetrics


def main():
    trainer_state = TRAINER.create_enrollment()

    # starts at stage_1_warmup
    current_stage = trainer_state.stage
    print(f"Current stage: {current_stage.name}")  # stage_1_warmup

    metrics = DynamicForagingMetrics(
        total_sessions=1,
        consecutive_sessions_at_current_stage=1,
        unignored_trials_per_session=[250],
        foraging_efficiency_per_session=[0.65],
        stage_name="stage_1_warmup",
    )

    # evaluate
    new_trainer_state = TRAINER.evaluate(trainer_state, metrics)
    print(f"Next stage: {new_trainer_state.stage.name}")  # stage_2, since finished_trials >= 200 and efficiency >= 0.6

    # save models
    trainer_state_path = Path(r".\local\trainer_state.json")
    trainer_state_path.write_text(new_trainer_state.model_dump_json(indent=2))

    metrics_path = Path(r".\local\metrics.json")
    metrics_path.write_text(metrics.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
