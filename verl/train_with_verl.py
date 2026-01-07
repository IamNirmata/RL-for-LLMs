# train_with_verl.py
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "grpo"], required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        "data.train_files=data/toy_train.parquet",
        "data.val_files=data/toy_val.parquet",
        f"actor_rollout_ref.model.path={args.model}",
        "custom_reward_function.path=rewards/my_reward.py",
    ]

    if args.algo == "grpo":
        cmd += [
            "algorithm.adv_estimator=grpo",
            "actor_rollout_ref.rollout.n=4",
            "actor_rollout_ref.actor.use_kl_loss=True",
        ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
