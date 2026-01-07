# verl PPO / GRPO 101 (Step-by-Step)

This package teaches you how to train a small LLM using PPO and GRPO with **verl**.

## What you will learn
1. How to prepare a parquet dataset for RL
2. How to write a custom reward function
3. How PPO works in practice
4. How GRPO differs from PPO (no critic, group-based baseline)
5. How to switch PPO → GRPO with minimal config changes

## Order of execution
1. Install verl
2. Run prepare_toy_parquet.py
3. Run train_with_verl.py with --algo ppo
4. Run train_with_verl.py with --algo grpo

See comments inside each Python file for detailed explanations.
