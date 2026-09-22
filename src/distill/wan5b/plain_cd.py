#!/usr/bin/env python3
"""Native bidirectional TI2V 5B conditioning, schedules and evaluation helpers."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

for root in Path(__file__).resolve().parents:
    if (root / "diffsynth").is_dir():
        sys.path.insert(0, str(root))
        break

CONTRACT = "plain_ti2v_5b_sixview_bidirectional_cd_v1"
ACTIONS = {"move up", "move down", "move left", "move right", "move forth", "move back"}


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_safetensors(tensors, path, metadata=None):
    from safetensors.torch import save_file
    save_file(tensors, str(path), metadata=metadata)


def write_video(frames, path, fps):
    from diffsynth.utils.data import VideoData, save_video
    save_video(frames, str(path), fps=fps, quality=5)
    if len(VideoData(str(path))) != len(frames):
        raise RuntimeError('Encoded video frame count mismatch')


def validate(cfg):
    from safetensors import safe_open
    if cfg["contract"] != CONTRACT:
        raise ValueError("Not a plain six-view CD recipe")
    with safe_open(cfg["teacher_checkpoint"], framework="pt", device="cpu") as f:
        if f.get_slice("patch_embedding.weight").get_shape() != [3072, 48, 1, 2, 2]:
            raise ValueError("Teacher must be the 48-channel plain TI2V 5B SFT; control/MoE checkpoints are incompatible")
        if any("control_moe" in k or "arm_action" in k for k in f.keys()):
            raise ValueError("Three-modality teacher is not navigation SFT")
    if cfg["num_frames"] != 21 or cfg["height"] * 3 != cfg["width"] * 2:
        raise ValueError("Six-view contract: 21 RGB frames and native 2:3 aspect ratio")
    if cfg["height"] % 16 or cfg["width"] % 16:
        raise ValueError("Spatial dimensions must be divisible by 16")
    if cfg["teacher_guidance"] != 1.0 or cfg["negative_prompt"] != "":
        raise ValueError("This recipe preserves the validated navigation SFT CFG=1, empty-negative contract")
    if cfg["cd_grid_steps"] < 2 or cfg["sigma_shift"] <= 0 or not 0 < cfg["ema_decay"] < 1:
        raise ValueError("Invalid CD schedule or EMA")
    if cfg["inference_steps"] < 1 or cfg["cd_grid_steps"] % cfg["inference_steps"]:
        raise ValueError("Inference steps must divide the CD grid so every inference sigma was trained")
    if Path(cfg["output_dir"]).resolve() in Path(cfg["teacher_checkpoint"]).resolve().parents:
        raise ValueError("Write CD to a new output directory, not the teacher directory")
    records = []
    with open(cfg["metadata"]) as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("prompt") not in ACTIONS or "video" not in row:
                raise ValueError(f"Row {index}: expected raw six-view video and one of six action prompts")
            if row.get("label_action_name", row["prompt"].replace(" ", "_")) != row["prompt"].replace(" ", "_"):
                raise ValueError(f"Row {index}: prompt/action-label mismatch")
            row = dict(row, dataset_index=index)
            records.append(row)
    if not records:
        raise ValueError("Empty navigation dataset")
    return records


def load_pipeline(cfg, device):
    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from safetensors.torch import load_file
    root = Path(cfg["model_root"])
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[
            ModelConfig(path=[str(root / f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors") for i in (1, 2, 3)]),
            ModelConfig(path=str(root / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=str(root / "Wan2.2_VAE.pth")),
        ], tokenizer_config=ModelConfig(path=str(root / "google/umt5-xxl")),
    )
    pipe.dit.load_state_dict(load_file(cfg["teacher_checkpoint"]), strict=True)
    pipe.requires_grad_(False).eval()
    if not pipe.dit.fuse_vae_embedding_in_latents or not pipe.dit.seperated_timestep:
        raise ValueError("Expected native 5B first-frame fusion and separated timestep")
    return pipe


def grid(steps, shift, device="cpu"):
    import torch
    # Include sigma=0 as the exact identity boundary of the consistency target.
    raw = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device=device)
    return shift * raw / (1 + (shift - 1) * raw)


def clamp_first(x, first):
    import torch
    return torch.cat((first.to(device=x.device, dtype=x.dtype), x[:, :, 1:]), dim=2)


def velocity(model, x, sigma, context, first, checkpointing=False):
    import torch
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    param = next(model.parameters())
    x = clamp_first(x.to(param.device), first)
    # Keep noise arithmetic FP32; cast only at the native DiT call boundary.
    return model_fn_wan_video(
        dit=model, latents=x.to(param.dtype),
        timestep=torch.as_tensor(float(sigma) * 1000, device=param.device, dtype=param.dtype).reshape(1),
        context=context.to(device=param.device, dtype=param.dtype),
        fuse_vae_embedding_in_latents=True,
        use_gradient_checkpointing=checkpointing,
    ).float()


def x0_prediction(model, x, sigma, context, first, checkpointing=False):
    param = next(model.parameters())
    x = clamp_first(x.to(param.device).float(), first)
    if float(sigma) == 0:
        return x
    return clamp_first(x - float(sigma) * velocity(model, x, sigma, context, first, checkpointing), first)


class RawCases:
    def __init__(self, pipe, cfg):
        self.pipe, self.cfg = pipe, cfg
        self.contexts = {}
        self.cache = Path(cfg["cache_dir"])
        self.cache.mkdir(parents=True, exist_ok=True)

    def get(self, row):
        import torch
        from safetensors.torch import load_file
        from diffsynth.utils.data import VideoData
        from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
        cfg, pipe = self.cfg, self.pipe
        vae_stamp = fingerprint(Path(cfg["model_root"]) / "Wan2.2_VAE.pth")
        text_stamp = fingerprint(Path(cfg["model_root"]) / "models_t5_umt5-xxl-enc-bf16.pth")
        identity = {"source": fingerprint(row["video"]), "vae": vae_stamp, "text_encoder": text_stamp,
                    "height": cfg["height"], "width": cfg["width"], "frames": cfg["num_frames"],
                    "prompt": row["prompt"], "contract": CONTRACT}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + ".safetensors")
        if path.is_file():
            return load_file(str(path))
        video = VideoData(row["video"])
        if len(video) != cfg["num_frames"]:
            raise ValueError(f"Expected 21 frames, got {len(video)}: {row['video']}")
        if video[0].width * 2 != video[0].height * 3:
            raise ValueError("Source six-view aspect ratio mismatch; do not stretch or crop view panels")
        frames = [video[i].resize((cfg["width"], cfg["height"])) for i in range(len(video))]
        with torch.no_grad():
            # Encode the observed first frame independently, exactly as native TI2V inference.
            first_image = pipe.preprocess_image(frames[0]).transpose(0, 1)
            first = pipe.vae.encode([first_image], device=pipe.device, tiled=False)
            clean = pipe.vae.encode(pipe.preprocess_video(frames), device=pipe.device, tiled=False)
            if row["prompt"] not in self.contexts:
                self.contexts[row["prompt"]] = WanVideoUnit_PromptEmbedder().encode_prompt(pipe, row["prompt"]).cpu()
            case = {"clean": clean.cpu(), "first": first.cpu(), "context": self.contexts[row["prompt"]]}
        expected = (1, 48, 6, cfg["height"] // 16, cfg["width"] // 16)
        if tuple(case["clean"].shape) != expected:
            raise ValueError(f"Latent contract mismatch: {case['clean'].shape} != {expected}")
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        write_safetensors({k: v.contiguous() for k, v in case.items()}, temporary, metadata={"identity": json.dumps(identity)})
        os.replace(temporary, path)
        return case


def evaluate(cfg, rows, args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    device = cfg["student_device"]
    pipe = load_pipeline(cfg, device)
    mode = args.eval_mode
    if mode == "student":
        if not args.checkpoint:
            raise ValueError("Student evaluation requires --checkpoint")
        with safe_open(args.checkpoint, framework="pt", device="cpu") as f:
            if (f.metadata() or {}).get("contract") != CONTRACT:
                raise ValueError("Checkpoint was not trained with this plain-navigation CD contract")
        trained_cfg = json.loads(Path(args.checkpoint).with_name("config.json").read_text())
        for key in ("height", "width", "num_frames", "sigma_shift", "teacher_checkpoint", "teacher_guidance", "negative_prompt"):
            # Public inference initializes directly from the selected student.
            if key == "teacher_checkpoint" and getattr(args, "inference_only", False):
                continue
            if trained_cfg[key] != cfg[key]:
                raise ValueError(f"Inference/training contract mismatch: {key}")
        pipe.dit.load_state_dict(load_file(args.checkpoint), strict=True)
    out = Path(args.eval_output or (Path(cfg["output_dir"]) / f"eval_{mode}"))
    out.mkdir(parents=True, exist_ok=True)
    cases = RawCases(pipe, cfg)
    steps = 50 if mode == "teacher" else cfg["inference_steps"]
    sigmas = grid(steps, cfg["sigma_shift"])
    manifest = []
    with torch.no_grad():
        for row in rows:
            case = cases.get(row)
            # Only the first frame, prompt, and output shape enter the rollout.
            generator = torch.Generator(device="cpu").manual_seed(cfg["seed"] + row["dataset_index"])
            x = torch.randn(case["clean"].shape, generator=generator, dtype=torch.bfloat16).to(device).float()
            x = clamp_first(x, case["first"])
            for i in range(steps):
                sigma, following = float(sigmas[i]), float(sigmas[i + 1])
                if mode == "teacher":
                    x = clamp_first(x - (sigma - following) * velocity(pipe.dit, x, sigma, case["context"], case["first"]), case["first"])
                else:
                    clean = x0_prediction(pipe.dit, x, sigma, case["context"], case["first"])
                    if following:
                        noise = torch.randn(x.shape, generator=generator, dtype=torch.bfloat16).to(device).float()
                        x = clamp_first((1 - following) * clean + following * noise, case["first"])
                    else:
                        x = clean
            decoded = pipe.vae.decode(x.to(torch.bfloat16), device=device, tiled=False)
            path = out / f"idx{row['dataset_index']:06d}_{row['prompt'].replace(' ', '_')}.mp4"
            write_video(pipe.vae_output_to_video(decoded), path, cfg["fps"])
            manifest.append({"index": row["dataset_index"], "prompt": row["prompt"], "source": row["video"],
                             "prediction": str(path), "checkpoint": args.checkpoint if mode == "student" else cfg["teacher_checkpoint"],
                             "contract": CONTRACT, "height": cfg["height"], "width": cfg["width"], "frames": cfg["num_frames"],
                             "sigmas": sigmas.tolist(), "seed": cfg["seed"] + row["dataset_index"], "mode": mode})
            print(json.dumps(manifest[-1]), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
