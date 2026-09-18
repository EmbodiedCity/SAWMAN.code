"""Native bidirectional Fun-InP CD: 21 RGB frames, CLIP plus masked-video y.

Unlike TI2V 5B, all six latent positions are noisy; the observation enters via y.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

for root in Path(__file__).resolve().parents:
    if (root / 'diffsynth').is_dir():
        sys.path.insert(0, str(root))
        break

CONTRACT = 'native_fun_inp_1p3b_sixview_bidirectional_cd_v1'
ACTIONS = {'move up', 'move down', 'move left', 'move right', 'move forth', 'move back'}

def fingerprint(path):
    p = Path(path).resolve(); s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)

def write_safetensors(tensors, path, metadata=None):
    from safetensors.torch import save_file
    save_file(tensors, str(path), metadata=metadata)


def write_video(frames, path, fps):
    from diffsynth.utils.data import VideoData, save_video
    save_video(frames, str(path), fps=fps, quality=5)
    if len(VideoData(str(path))) != len(frames):
        raise RuntimeError('Encoded video frame count mismatch')


def grid(steps, shift, device='cpu'):
    import torch
    s = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device=device)
    return shift * s / (1 + (shift - 1) * s)

def validate(cfg):
    from safetensors import safe_open
    if cfg['contract'] != CONTRACT or cfg['training_stage'] not in {'stage2','stage3'}:
        raise ValueError('Expected native Fun-InP Stage2/Stage3 recipe')
    if (cfg['num_frames'], cfg['height'], cfg['width']) != (21, 640, 960):
        raise ValueError('Expected native six-view 21x640x960 RGB')
    if cfg['inference_steps'] < 1 or cfg['cd_grid_steps'] % cfg['inference_steps']:
        raise ValueError('Inference sigma grid must be a subset of training grid')
    if cfg['teacher_guidance'] < 1 or not 0 < cfg['ema_decay'] < 1:
        raise ValueError('Invalid guidance or EMA')
    with safe_open(cfg['teacher_checkpoint'], framework='pt', device='cpu') as f:
        if f.get_slice('patch_embedding.weight').get_shape() != [1536, 36, 1, 2, 2]:
            raise ValueError('Expected native 36-channel Fun-InP SFT')
    records = []
    for index, line in enumerate(open(cfg['metadata'])):
        if not line.strip():
            continue
        row = json.loads(line)
        if row['prompt'] not in ACTIONS or row.get('video_frame_count', 21) != 21:
            raise ValueError(f'Bad navigation row {index}')
        if row.get('label_action_name', row['prompt'].replace(' ', '_')) != row['prompt'].replace(' ', '_'):
            raise ValueError(f'Action label mismatch {index}')
        records.append(dict(row, dataset_index=index))
    return records

def load_pipeline(cfg, device):
    import torch
    from safetensors.torch import load_file
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    p = Path(cfg['model_root'])
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=str(p / name)) for name in [
            'diffusion_pytorch_model.safetensors', 'models_t5_umt5-xxl-enc-bf16.pth',
            'Wan2.1_VAE.pth', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth']],
        tokenizer_config=ModelConfig(path=str(p / 'google/umt5-xxl')))
    pipe.dit.load_state_dict(load_file(cfg['teacher_checkpoint']), strict=True)
    pipe.requires_grad_(False).eval()
    if (not pipe.dit.require_clip_embedding or not pipe.dit.require_vae_embedding
            or pipe.dit.fuse_vae_embedding_in_latents):
        raise ValueError('Native Fun-InP CLIP/y conditioning required')
    return pipe

class RawCases:
    def __init__(self, pipe, cfg):
        self.pipe, self.cfg = pipe, cfg
        self.cache = Path(cfg['cache_dir']); self.cache.mkdir(parents=True, exist_ok=True)
        self.contexts = {}

    def check(self, case):
        expected = {'clean': (1,16,6,80,120), 'y': (1,20,6,80,120),
                    'clip': (1,257,1280), 'context': (1,512,4096), 'negative_context': (1,512,4096)}
        for key, shape in expected.items():
            if tuple(case[key].shape) != shape:
                raise ValueError(f'{key} shape {tuple(case[key].shape)} != {shape}')
        import torch
        if not all(torch.isfinite(x).all() for x in case.values()):
            raise ValueError('Nonfinite cached conditioning')
        if not (case['y'][:,:4,:1] == 1).all() or not (case['y'][:,:4,1:] == 0).all():
            raise ValueError('Fun-InP first-frame mask mismatch')
        return case

    def get(self, row):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        from diffsynth.utils.data import VideoData
        from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
        pipe, cfg = self.pipe, self.cfg
        identity = dict(source=fingerprint(row['video']), contract=CONTRACT,
                        shape=[21,640,960], prompt=row['prompt'], negative=cfg['negative_prompt'],
                        encoders=[fingerprint(Path(cfg['model_root']) / n) for n in [
                            'Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
                            'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth']], dtype='bfloat16')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + '.safetensors')
        if path.exists():
            with safe_open(str(path), framework='pt', device='cpu') as f:
                if json.loads(f.metadata()['identity']) != identity:
                    raise ValueError('Cache identity mismatch')
            return self.check(load_file(str(path)))
        video = VideoData(row['video'])
        if len(video) != 21 or video[0].size != (960,640):
            raise ValueError(f'Raw sample is not 21 frames at 960x640: {row["video"]}')
        frames = [video[i] for i in range(21)]
        with torch.no_grad():
            clean = pipe.vae.encode(pipe.preprocess_video(frames), device=pipe.device, tiled=False)
            image = pipe.preprocess_image(frames[0]).to(pipe.device)
            clip = pipe.image_encoder.encode_image([image])
            # Identical to native WanVideoUnit_ImageEmbedderVAE for first-frame-only conditioning.
            observed = torch.cat([image.transpose(0,1), torch.zeros(3,20,640,960,device=pipe.device)], dim=1)
            encoded = pipe.vae.encode([observed.to(pipe.torch_dtype)], device=pipe.device, tiled=False)
            mask = torch.zeros(1,4,6,80,120, device=pipe.device, dtype=pipe.torch_dtype)
            mask[:,:,:1] = 1
            y = torch.cat([mask, encoded], dim=1)
            for prompt in [row['prompt'], cfg['negative_prompt']]:
                if prompt not in self.contexts:
                    self.contexts[prompt] = WanVideoUnit_PromptEmbedder().encode_prompt(pipe, prompt).cpu()
            case = dict(clean=clean.cpu(), clip=clip.cpu(), y=y.cpu(),
                        context=self.contexts[row['prompt']], negative_context=self.contexts[cfg['negative_prompt']])
        self.check(case)
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        write_safetensors({k:v.contiguous() for k,v in case.items()}, temporary, {'identity':json.dumps(identity)})
        os.replace(temporary, path)
        return case
