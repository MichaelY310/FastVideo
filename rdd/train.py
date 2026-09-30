# SPDX-License-Identifier: Apache-2.0
"""Minimal guarded entrypoint using FastVideo's existing modular trainer."""

import argparse
import faulthandler
import os
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    if not os.environ.get("RDD_JOB_TOKEN") or not os.environ.get("RDD_DEADLINE_UNIX"):
        parser.error("launch via python -m rdd.deadline; an absolute deadline is mandatory")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train import Trainer
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.checkpoint import CheckpointConfig, CheckpointManager
    from rdd.finetune import DeadlineReached
    import torch.distributed as dist

    cfg = load_run_config(args.config, overrides=overrides)
    tc = cfg.training
    maybe_init_distributed_environment_and_model_parallel(tc.distributed.tp_size, tc.distributed.sp_size)
    succeeded = False
    try:
        _, method, loader, start = build_from_config(cfg)
        print(f"RDD rank={os.environ.get('RANK')} model/data/optimizer ready", flush=True)
        trainer = Trainer(tc, config=cfg.resolved_config(), callback_configs=cfg.callbacks)
        print(f"RDD rank={os.environ.get('RANK')} trainer ready", flush=True)
        manager = CheckpointManager(method=method, dataloader=loader, output_dir=tc.checkpoint.output_dir,
                                    config=CheckpointConfig(save_steps=tc.checkpoint.training_state_checkpointing_steps,
                                                            keep_last=tc.checkpoint.checkpoints_total_limit),
                                    callbacks=trainer.callbacks, raw_config=cfg.raw)
        try:
            trainer.run(method, dataloader=loader, max_steps=tc.loop.max_train_steps,
                        start_step=start, checkpoint_manager=manager)
        except DeadlineReached:
            manager.save_final(method.last_iteration)
            print(f"RDD saved at step {method.last_iteration}; exiting before hard deadline", flush=True)
        succeeded = True
    except BaseException:
        traceback.print_exc()
        # Do not hide an initialization error in a collective destroy call
        # while other ranks are still building modules. torchrun reaps peers.
        raise
    finally:
        if succeeded and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
