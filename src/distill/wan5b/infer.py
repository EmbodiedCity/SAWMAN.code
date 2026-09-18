import argparse,json
from pathlib import Path
import plain_cd as common

def main():
    p=argparse.ArgumentParser(description="Native 5B student inference")
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=4);p.add_argument('--limit',type=int,default=2)
    a=p.parse_args();cfg=json.loads(Path(a.checkpoint).with_name('config.json').read_text())
    cfg['inference_steps']=a.steps;cfg['student_device']='cuda:0'
    rows=common.validate(cfg)[:a.limit];a.eval_mode='student';a.eval_output=a.output
    common.evaluate(cfg,rows,a)
if __name__=='__main__':main()
