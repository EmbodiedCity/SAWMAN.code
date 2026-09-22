"""Native full-video Fun-InP student evaluation; no causal or latent-prefix path."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
import plain_cd as common
from distributed_train import NativeWan, x0

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--limit', type=int, default=2)
    p.add_argument('--model-root', help='Matching upstream base model directory')
    p.add_argument('--metadata', help='Evaluation JSONL with 21-frame six-view videos')
    args = p.parse_args()
    if args.limit < 1:
        p.error('--limit must be positive')
    checkpoint = Path(args.checkpoint)
    cfg = json.loads((checkpoint.parent/'config.json').read_text())
    # Inference needs the released DiT, not the private SFT teacher.
    cfg['teacher_checkpoint'] = str(checkpoint)
    if args.model_root: cfg['model_root'] = args.model_root
    if args.metadata: cfg['metadata'] = args.metadata
    cfg['inference_steps'] = args.steps
    if args.steps < 1 or cfg['cd_grid_steps'] % args.steps:
        raise ValueError('Invalid inference grid')
    with safe_open(str(checkpoint),framework='pt',device='cpu') as f:
        if (f.metadata() or {}).get('contract') != common.CONTRACT:
            raise ValueError('Wrong model contract')
    torch.set_num_threads(4)
    rows = common.validate(cfg)[:args.limit]
    pipe = common.load_pipeline(cfg,'cuda:0')
    pipe.dit.load_state_dict(load_file(str(checkpoint)),strict=True)
    model = NativeWan(pipe.dit).eval()
    cases = common.RawCases(pipe,cfg)
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    manifest = []
    for row in rows:
        case = {k:v.cuda() for k,v in cases.get(row).items()}
        g = torch.Generator(device='cpu').manual_seed(cfg['seed']+row['dataset_index'])
        values = torch.randn(case['clean'].shape,dtype=torch.bfloat16,generator=g).cuda().float()
        sigmas = common.grid(args.steps,cfg['sigma_shift'])
        for i in range(args.steps):
            clean = x0(model,values,float(sigmas[i]),case)
            following = float(sigmas[i+1])
            if following:
                noise = torch.randn(values.shape,dtype=torch.bfloat16,generator=g).cuda().float()
                values = (1-following)*clean + following*noise
            else:
                values = clean
        decoded = pipe.vae.decode(values.to(torch.bfloat16),device='cuda:0',tiled=False)
        path = out/f"idx{row['dataset_index']:06d}_{row['prompt'].replace(' ','_')}.mp4"
        common.write_video(pipe.vae_output_to_video(decoded),path,cfg['fps'])
        manifest.append(dict(index=row['dataset_index'],prompt=row['prompt'],source=row['video'],prediction=str(path),
                             checkpoint=str(checkpoint),steps=args.steps,frames=21,height=640,width=960,
                             seed=cfg['seed']+row['dataset_index'],sigmas=sigmas.tolist()))
        print(json.dumps(manifest[-1]),flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
