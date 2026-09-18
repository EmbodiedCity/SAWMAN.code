#!/usr/bin/env python3
"""Eight-rank FSDP native navigation CD, followed by bidirectional DMD.

All networks retain the SFT's bidirectional attention and native Fun-InP CLIP and masked-video y
conditioning. Rank-local FP32 shards are optimized directly; BF16 is only compute.
"""
import argparse
import copy
from datetime import timedelta
from functools import partial
import gc
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,
    FullStateDictConfig, StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors import safe_open
from safetensors.torch import load_file

import plain_cd as common
from diffsynth.models.wan_video_dit import DiTBlock
from diffsynth.pipelines.wan_video import model_fn_wan_video


class NativeWan(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, timestep, context, clip, y, checkpointing=False):
        return model_fn_wan_video(
            dit=self.dit, latents=x.to(torch.bfloat16), timestep=timestep.to(torch.bfloat16),
            context=context.to(torch.bfloat16), clip_feature=clip.to(torch.bfloat16),
            y=y.to(torch.bfloat16), fuse_vae_embedding_in_latents=False,
            use_gradient_checkpointing=checkpointing,
        ).float()


def velocity(model, x, sigma, case, checkpointing=False):
    t = torch.tensor([float(sigma) * 1000], device=x.device, dtype=torch.bfloat16)
    return model(x, t, case["context"], case["clip"], case["y"], checkpointing)


def x0(model, x, sigma, case, checkpointing=False):
    x = x.float()
    if float(sigma) == 0:
        return x
    return x - float(sigma) * velocity(model, x, sigma, case, checkpointing)


def wrap(base, device, trainable, checkpoint=None):
    dit = copy.deepcopy(base).float()
    if checkpoint:
        dit.load_state_dict(load_file(str(checkpoint)), strict=True)
    net = NativeWan(dit).requires_grad_(trainable).train(trainable)
    return FSDP(
        net, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={DiTBlock}),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.bfloat16),
        use_orig_params=False, limit_all_gathers=True,
    )


@torch.no_grad()
def update_ema(ema, student, decay):
    for avg, parameter in zip(ema.parameters(), student.parameters()):
        if avg.shape != parameter.shape or avg.dtype != torch.float32 or parameter.dtype != torch.float32:
            raise RuntimeError("EMA and student must have matching FP32 FSDP shards")
        avg.lerp_(parameter.detach(), 1 - decay)


def teacher_velocity(model, x, sigma, case, cfg):
    positive = velocity(model, x, sigma, case)
    if cfg['teacher_guidance'] == 1:
        return positive
    negative = velocity(model, x, sigma, dict(case, context=case['negative_context']))
    return negative + cfg['teacher_guidance'] * (positive - negative)


def stage2_step(student, teacher, ema, optimizer, case, cfg, step):
    sigmas = common.grid(cfg["cd_grid_steps"], cfg["sigma_shift"])
    # Different per-rank zero-boundary branches would deadlock FSDP collectives.
    j = random.Random(cfg["seed"] + step * 1009).randrange(cfg["cd_grid_steps"])
    sigma, following = float(sigmas[j]), float(sigmas[j + 1])
    clean = case["clean"].float()
    x = (1 - sigma) * clean + sigma * torch.randn_like(clean)
    with torch.no_grad():
        endpoint = x - (sigma - following) * teacher_velocity(teacher, x, sigma, case, cfg)
        target = x0(ema, endpoint, following, case)
    predicted = x0(student, x, sigma, case, checkpointing=True)
    loss = (predicted - target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite CD loss")
    loss.backward()
    norm = student.clip_grad_norm_(cfg["max_grad_norm"])
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite CD gradient")
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    update_ema(ema, student, cfg["ema_decay"])
    return {"cd_loss": float(loss.detach()), "grad_norm": float(norm), "sigma": sigma,
            "next_sigma": following, "generator_updated": 1.0}


def dmd_gradient(generated, real_x0, fake_x0):
    # -(real score - fake score), expressed in x0 space. Normalize per sample.
    scale = (generated - real_x0).abs().mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-3)
    gradient = (fake_x0 - real_x0) / scale
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite DMD gradient")
    return gradient


def rollout(student, case, sigmas, exit_index, with_grad):
    x = torch.randn_like(case["clean"], dtype=torch.float32)
    for i in range(exit_index + 1):
        # Earlier denoising steps are detached; all ranks use the same exit index.
        with torch.set_grad_enabled(with_grad and i == exit_index):
            clean = x0(student, x, float(sigmas[i]), case, checkpointing=with_grad and i == exit_index)
        if i == exit_index:
            return clean
        following = float(sigmas[i + 1])
        x = (1 - following) * clean.detach() + following * torch.randn_like(x)
    raise AssertionError("Empty rollout")


def stage3_step(student, teacher, ema, fake, optimizer, fake_optimizer, case, cfg, step):
    draw = random.Random(cfg["seed"] + step * 1009)
    update_generator = (step - 1) % cfg["critic_updates_per_generator"] == 0
    sigmas = common.grid(cfg["inference_steps"], cfg["sigma_shift"])
    exit_index = draw.randrange(cfg["inference_steps"])
    generated = rollout(student, case, sigmas, exit_index, update_generator)
    # Score training spans the shifted noise distribution, with finite endpoints.
    raw_sigma = draw.uniform(.02, .98)
    sigma = cfg["sigma_shift"] * raw_sigma / (1 + (cfg["sigma_shift"] - 1) * raw_sigma)
    noise = torch.randn_like(generated)
    noisy = (1 - sigma) * generated.detach() + sigma * noise
    generator_loss = 0.0; generator_norm = 0.0
    if update_generator:
        with torch.no_grad():
            real_x0 = noisy - sigma * teacher_velocity(teacher, noisy, sigma, case, cfg)
            fake_x0 = x0(fake, noisy, sigma, case)
            grad = dmd_gradient(generated.detach(), real_x0, fake_x0)
            target = generated.detach() - grad
        loss = .5 * (generated - target).square().mean()
        loss.backward()
        norm = student.clip_grad_norm_(cfg["max_grad_norm"])
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite DMD generator update")
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        update_ema(ema, student, cfg["ema_decay"])
        generator_loss, generator_norm = float(loss.detach()), float(norm)
    # Fake score learns the *generated* distribution, never the dataset future.
    fake_velocity = velocity(fake, noisy, sigma, case, checkpointing=True)
    flow_target = noise - generated.detach()
    fake_loss = (fake_velocity - flow_target).square().mean()
    fake_loss.backward()
    fake_norm = fake.clip_grad_norm_(cfg["max_grad_norm"])
    if not torch.isfinite(fake_loss) or not torch.isfinite(fake_norm):
        raise FloatingPointError("Nonfinite fake-score update")
    fake_optimizer.step(); fake_optimizer.zero_grad(set_to_none=True)
    return {"dmd_loss": generator_loss, "grad_norm": generator_norm, "fake_loss": float(fake_loss.detach()),
            "fake_grad_norm": float(fake_norm), "sigma": sigma, "exit_index": float(exit_index),
            "generator_updated": float(update_generator)}


def local_weights(model):
    return [p.detach().cpu().clone() for p in model.parameters()]


def restore_local(model, values):
    parameters = list(model.parameters())
    if len(parameters) != len(values):
        raise ValueError("FSDP partition changed")
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            if parameter.shape != value.shape:
                raise ValueError("FSDP shard shape changed")
            parameter.copy_(value)


@torch.no_grad()
def evaluate_saved(destination, cfg, ema, teacher, pipe, cases, rows, device):
    """Fixed six-action teacher/student comparison, collectively across all ranks."""
    representatives = {}
    for row in rows:
        representatives.setdefault(row["prompt"], row)
    if set(representatives) != common.ACTIONS:
        raise ValueError("Checkpoint quality evaluation requires all six navigation actions")
    selected = [representatives[action] for action in sorted(common.ACTIONS)]
    rank = dist.get_rank()
    row = selected[rank % 6]
    case = {k: v.to(device) for k, v in cases.get(row).items()}
    out = destination / "evaluation" / f"rank{rank}_{row['prompt'].replace(' ', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"] + row["dataset_index"]
    images = {}
    for kind, model, steps in (("teacher", teacher, 50), ("student", ema, cfg["inference_steps"])):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = torch.randn(case["clean"].shape, dtype=torch.bfloat16, generator=generator).to(device).float()
        sigmas = common.grid(steps, cfg["sigma_shift"])
        for i in range(steps):
            sigma, following = float(sigmas[i]), float(sigmas[i + 1])
            if kind == "teacher":
                values = values - (sigma - following) * teacher_velocity(model, values, sigma, case, cfg)
            else:
                clean = x0(model, values, sigma, case)
                if following:
                    noise = torch.randn(values.shape, dtype=torch.bfloat16, generator=generator).to(device).float()
                    values = (1 - following) * clean + following * noise
                else:
                    values = clean
        decoded = pipe.vae.decode(values.to(torch.bfloat16), device=device, tiled=False)
        common.write_video(pipe.vae_output_to_video(decoded), out / f"{kind}.mp4", cfg["fps"])
        images[kind] = ((decoded.float() + 1) / 2).clamp(0, 1).cpu()
    decoded_gt = pipe.vae.decode(case["clean"].to(torch.bfloat16), device=device, tiled=False)
    common.write_video(pipe.vae_output_to_video(decoded_gt), out / "gt_vae_reconstruction.mp4", cfg["fps"])
    gt = ((decoded_gt.float() + 1) / 2).clamp(0, 1).cpu()
    metrics = {"dataset_index": row["dataset_index"], "prompt": row["prompt"], "seed": seed, "rank": rank,
               "output": str(out), "reference": "VAE reconstruction of GT; metrics exclude observed RGB frame"}
    gt_motion = float((gt[:, :, 2:] - gt[:, :, 1:-1]).abs().mean())
    for kind, pixels in images.items():
        mse = (pixels[:, :, 1:] - gt[:, :, 1:]).square().mean().clamp_min(1e-12)
        motion = float((pixels[:, :, 2:] - pixels[:, :, 1:-1]).abs().mean())
        metrics[kind + "_psnr"] = float(-10 * torch.log10(mse))
        metrics[kind + "_motion_ratio"] = motion / max(gt_motion, 1e-8)
    metrics["low_motion"] = metrics["student_motion_ratio"] < .15
    all_metrics = [None] * 8
    dist.all_gather_object(all_metrics, metrics)
    if rank == 0:
        summary = {"complete": True, "stage": cfg["training_stage"], "teacher": cfg["teacher_checkpoint"],
                   "metadata": cfg["metadata"], "all_six_actions": True,
                   "low_motion_cases": sum(item["low_motion"] for item in all_metrics[:6]), "samples": all_metrics}
        (destination / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"evaluation": str(destination / "evaluation_summary.json"),
                          "low_motion_cases": summary["low_motion_cases"]}), flush=True)
    dist.barrier()


def save_checkpoint(out, step, cfg, student, ema, optimizer, fake, fake_optimizer):
    rank = dist.get_rank()
    destination = out / f"checkpoint-{step}"
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    started = time.perf_counter()
    for name, model in (("diffusion_pytorch_model", student), ("diffusion_pytorch_model_ema", ema)):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = model.state_dict()
        if rank == 0:
            state = {k.removeprefix("dit."): v.to(torch.bfloat16).contiguous() for k, v in state.items()}
            common.write_safetensors(state, destination / f"{name}.safetensors",
                                     metadata={"contract": common.CONTRACT, "training_stage": cfg["training_stage"],
                                               "step": str(step), "teacher": json.dumps(common.fingerprint(cfg["teacher_checkpoint"]))})
            from safetensors import safe_open
            with safe_open(str(cfg['teacher_checkpoint']), framework='pt', device='cpu') as source:
                expected = {k: source.get_slice(k).get_shape() for k in source.keys()}
            if set(state) != set(expected) or any(list(state[k].shape) != expected[k] for k in state):
                raise RuntimeError('Incomplete native student/EMA export')
        del state
        gc.collect(); dist.barrier()
    if cfg.get("save_optimizer", True):
        state = {"rank": rank, "world_size": dist.get_world_size(), "step": step,
                 "student": local_weights(student), "ema": local_weights(ema), "optimizer": optimizer.state_dict(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                 "fake": local_weights(fake) if fake is not None else None,
                 "fake_optimizer": fake_optimizer.state_dict() if fake_optimizer is not None else None}
        torch.save(state, destination / f"training_rank{rank}.pt")
        del state
    dist.barrier()
    if rank == 0:
        (destination / "config.json").write_text(json.dumps(cfg, indent=2))
        (destination / ".complete").write_text(str(step))
        print(json.dumps({"checkpoint": str(destination), "save_seconds": time.perf_counter() - started}), flush=True)
    dist.barrier()


def verify_generator(cfg, allow_pending=False):
    path = Path(cfg["generator_checkpoint"])
    if not path.exists() and allow_pending:
        return "pending Stage2 completion"
    if not (path.parent / ".complete").exists():
        raise ValueError("Stage3 requires a completed Stage2 checkpoint")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        if metadata.get("contract") != common.CONTRACT or metadata.get("training_stage", "stage2") != "stage2":
            raise ValueError("Stage3 generator is not compatible navigation Stage2")
    trained = json.loads((path.parent / "config.json").read_text())
    for key in ("teacher_checkpoint", "metadata", "height", "width", "num_frames", "sigma_shift", "cd_grid_steps", "teacher_guidance", "negative_prompt"):
        if cfg[key] != trained[key]:
            raise ValueError(f"Stage2/Stage3 mismatch: {key}")
    if cfg.get("require_stage2_evaluation", True):
        summary = json.loads((path.parent / "evaluation_summary.json").read_text())
        if not summary.get("complete") or not summary.get("all_six_actions") or summary.get("low_motion_cases", 6):
            raise ValueError("Stage2 six-action evaluation is incomplete or still detects low-motion collapse; inspect it before Stage3")
    if trained["cd_grid_steps"] % cfg["inference_steps"]:
        raise ValueError("DMD sampling nodes must belong to Stage2 grid")
    return "matched completed native Fun-InP Stage2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage2", "stage3"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--allow-pending-stage2", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int, help="Diagnostic stop; preserve config and test exact resume")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["training_stage"] != args.stage or cfg["world_size"] != 8:
        raise ValueError("Expected an eight-rank matching-stage recipe")
    rows = common.validate(cfg)
    if args.limit:
        rows = rows[:args.limit]
    if len(rows) < 8:
        raise ValueError("Need at least one distinct record per rank")
    status = verify_generator(cfg) if args.stage == "stage3" else "native bidirectional Fun-InP; teacher/student/EMA from SFT"
    if args.preflight:
        print(json.dumps({"rows": len(rows), "metadata": cfg["metadata"], "stage": args.stage, "initialization": status,
                          "shape": [cfg["num_frames"], cfg["height"], cfg["width"]], "world_size": 8,
                          "output": cfg["output_dir"], "max_steps": cfg["max_steps"], "save_every": cfg["save_every"]}), flush=True)
        return
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise ValueError("Launch with torchrun --nproc_per_node=8")
    rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.manual_seed(cfg["seed"] + rank)
    out = Path(cfg["output_dir"])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        lineage = {"teacher": common.fingerprint(cfg["teacher_checkpoint"]), "metadata": common.fingerprint(cfg["metadata"]),
                   "selected_rows": len(rows), "config": cfg}
        if (out / "lineage.json").exists() and json.loads((out / "lineage.json").read_text()) != lineage:
            raise ValueError("Output lineage mismatch; choose a new output directory")
        if list(out.glob("checkpoint-*/.complete")) and not args.resume:
            raise ValueError("Checkpoints exist: resume explicitly")
        (out / "lineage.json").write_text(json.dumps(lineage, indent=2))
        print(f"[initializing] {args.stage} eight-rank FSDP, full DiT, raw navigation data", flush=True)
    dist.barrier()
    pipe = common.load_pipeline(cfg, "cpu")
    base = pipe.dit
    pipe.dit = None
    initialization = cfg["generator_checkpoint"] if args.stage == "stage3" else None
    student = wrap(base, device, True, initialization)
    teacher = wrap(base, device, False)
    ema = wrap(base, device, False, initialization)
    fake = wrap(base, device, True) if args.stage == "stage3" else None
    del base; gc.collect()
    pipe.device = device
    pipe.vae.to(device); pipe.text_encoder.to(device); pipe.image_encoder.to(device)
    cases = common.RawCases(pipe, cfg)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["learning_rate"], betas=(0.9, .999), weight_decay=.01)
    fake_optimizer = torch.optim.AdamW(fake.parameters(), lr=cfg["critic_learning_rate"], betas=(0.9, .999), weight_decay=.01) if fake else None
    if any(p.dtype != torch.float32 for p in student.parameters()):
        raise RuntimeError('Optimizer master parameters must be FP32')
    first_step = 0
    if args.resume:
        checkpoint = Path(args.resume)
        if not (checkpoint / ".complete").exists() or json.loads((checkpoint / "config.json").read_text()) != cfg:
            raise ValueError("Incomplete checkpoint or resume-config mismatch")
        state = torch.load(checkpoint / f"training_rank{rank}.pt", map_location="cpu", weights_only=False)
        if state["rank"] != rank or state["world_size"] != 8:
            raise ValueError("Resume partition mismatch")
        restore_local(student, state["student"]); restore_local(ema, state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        if fake:
            restore_local(fake, state["fake"]); fake_optimizer.load_state_dict(state["fake_optimizer"])
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
        first_step = state["step"]; del state
    dist.barrier()
    if rank == 0:
        print(f"[ready] all 8 ranks initialized; resume_step={first_step}", flush=True)
    epoch = None; order = None
    for step in range(first_step + 1, cfg["max_steps"] + 1):
        started = time.perf_counter()
        position = (step - 1) * 8 + rank
        row = None
        for attempt in range(cfg.get("max_skip_attempts", 32)):
            candidate = position + attempt * 8
            current_epoch, offset = divmod(candidate, len(rows))
            if epoch != current_epoch:
                epoch = current_epoch; order = list(range(len(rows)))
                random.Random(cfg["seed"] + epoch).shuffle(order)
            row = rows[order[offset]]
            try:
                raw = cases.get(row)
                case = {k: v.to(device) for k, v in raw.items()}
                break
            except Exception as error:
                print(f"[data-error] rank={rank} index={row['dataset_index']} attempt={attempt + 1}: {error}", flush=True)
                if attempt + 1 == cfg.get("max_skip_attempts", 32):
                    raise
        if args.stage == "stage2":
            metrics = stage2_step(student, teacher, ema, optimizer, case, cfg, step)
        else:
            metrics = stage3_step(student, teacher, ema, fake, optimizer, fake_optimizer, case, cfg, step)
        values = torch.tensor(list(metrics.values()), dtype=torch.float64, device=device)
        dist.all_reduce(values); values /= 8
        metrics = dict(zip(metrics, values.cpu().tolist()))
        elapsed = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        indices = [None] * 8
        dist.all_gather_object(indices, row["dataset_index"])
        metrics.update(step=step, stage=args.stage, seconds=float(elapsed), samples_per_step=8, dataset_indices=indices)
        if rank == 0:
            with (out / "loss.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
        if step % cfg["save_every"] == 0 or step == cfg["max_steps"]:
            save_checkpoint(out, step, cfg, student, ema, optimizer, fake, fake_optimizer)
            if cfg.get("evaluate_on_save", True):
                evaluate_saved(out / f"checkpoint-{step}", cfg, ema, teacher, pipe, cases, rows, device)
        if args.stop_after and step >= args.stop_after:
            break
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
