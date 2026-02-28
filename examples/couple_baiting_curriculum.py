from aind_behavior_dynamic_foraging.curricula.coupled_baiting import TRAINER
from aind_behavior_dynamic_foraging.curricula.metrics import DynamicForagingMetrics


def main():
    trainer_state = TRAINER.create_enrollment()

    # starts at stage_1_warmup
    current_stage = trainer_state.stage
    print(f"Current stage: {current_stage.name}")  # stage_1_warmup

    metrics = DynamicForagingMetrics(
        session_total=1,
        session_at_current_stage=1,
        finished_trials=[250],
        foraging_efficiency=[0.65],
    )

    # evaluate
    new_trainer_state = TRAINER.evaluate(trainer_state, metrics)
    print(f"Next stage: {new_trainer_state.stage.name}")  # stage_2, since finished_trials >= 200 and efficiency >= 0.6


if __name__ == "__main__":
    main()
