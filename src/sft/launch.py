"""Build the same native SFT command without machine-specific storage wrappers."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def command(model,c):
 root=Path(c['model_root'])
 dit=[str(root/f'diffusion_pytorch_model-{i:05d}-of-00003.safetensors') for i in range(1,4)] if model=='wan5b' else str(root/'diffusion_pytorch_model.safetensors')
 paths=[dit,str(root/'models_t5_umt5-xxl-enc-bf16.pth'),str(root/('Wan2.2_VAE.pth' if model=='wan5b' else 'Wan2.1_VAE.pth'))]
 if model=='wan1p3b':paths.append(str(root/'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'))
 cmd=[sys.executable,'-m','accelerate.commands.launch','--config_file',f"configs/accelerate_{c['backend']}.yaml",'--num_processes',str(c['num_processes']),'src/sft/train.py']
 opts=dict(dataset_base_path='.',dataset_metadata_path=c['metadata'],num_frames=21,dataset_repeat=1,dataset_num_workers=8,model_paths=json.dumps(paths),tokenizer_path=str(root/'google/umt5-xxl'),learning_rate=c['learning_rate'],num_epochs=c['num_epochs'],save_steps=c['save_steps'],remove_prefix_in_ckpt='pipe.dit.',output_path=c['output_dir'],extra_inputs='input_image',gradient_accumulation_steps=1,trainable_models='dit',max_skip_attempts=32)
 for key,value in opts.items():cmd.extend(['--'+key,str(value)])
 cmd.extend(['--skip_failed_samples','--use_gradient_checkpointing' if c['gradient_checkpointing'] else '--force_disable_gradient_checkpointing'])
 if c['initialize_model_on_cpu']:cmd.append('--initialize_model_on_cpu')
 if c.get('resume_from_checkpoint'):cmd.extend(['--resume_from_checkpoint',c['resume_from_checkpoint']])
 return cmd

def main():
 p=argparse.ArgumentParser();p.add_argument('model',choices=['wan5b','wan1p3b']);p.add_argument('--config');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 c=json.loads(Path(a.config or f'configs/{a.model}_sft.json').read_text());cmd=command(a.model,c)
 if a.dry_run:print(json.dumps(cmd,indent=2));return
 Path(c['output_dir']).mkdir(parents=True,exist_ok=True)
 subprocess.run(cmd,check=True)
if __name__=='__main__':main()
