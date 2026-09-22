import argparse,json
from pathlib import Path
import plain_cd as common

def main():
    p=argparse.ArgumentParser(description="Native 5B student inference")
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--limit',type=int,default=2)
    p.add_argument('--model-root',help='Matching upstream base model directory')
    p.add_argument('--metadata',help='Evaluation JSONL with 21-frame six-view videos')
    a=p.parse_args()
    if a.limit < 1: p.error('--limit must be positive')
    cfg=json.loads(Path(a.checkpoint).with_name('config.json').read_text())
    cfg['teacher_checkpoint']=a.checkpoint
    if a.model_root: cfg['model_root']=a.model_root
    if a.metadata: cfg['metadata']=a.metadata
    a.inference_only=True
    cfg['inference_steps']=a.steps;cfg['student_device']='cuda:0'
    rows=common.validate(cfg)[:a.limit];a.eval_mode='student';a.eval_output=a.output
    common.evaluate(cfg,rows,a)
if __name__=='__main__':main()
