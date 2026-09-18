"""Convert chain CSV + six-view PNGs into 21-frame forward/reverse navigation clips."""
import argparse,csv,json,os
from pathlib import Path
ACTIONS={6:'move_forth',7:'move_back',8:'move_left',9:'move_right',10:'move_up',11:'move_down'}
INVERSE={6:7,7:6,8:9,9:8,10:11,11:10}

def segments(rows):
    # Every action contributes 20 new frames; include the preceding observed frame.
    for start in range(0,len(rows)-20,20):
        clip=rows[start:start+21]
        if [int(r['frame_id']) for r in clip]!=list(range(start,start+21)):
            raise ValueError('Chain has missing/nonconsecutive frames')
        action=int(clip[1]['action'])
        if action not in ACTIONS or any(int(r['action'])!=action or r['action_name']!=ACTIONS[action] for r in clip[1:]):
            raise ValueError('Action labels do not match a 20-frame action segment')
        yield start,action,clip

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',default='data/raw');p.add_argument('--output',default='data/videos');p.add_argument('--metadata',default='data/metadata.jsonl');p.add_argument('--no-reverse',action='store_true');a=p.parse_args()
    import imageio.v2 as imageio
    import numpy as np
    root=Path(a.raw_root);out=Path(a.output);meta=Path(a.metadata)
    if meta.exists():raise FileExistsError('Use a fresh metadata/output destination')
    out.mkdir(parents=True,exist_ok=True);records=[]
    for source in sorted(root.rglob('chain_*.csv')):
        with source.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(line for line in f if not line.startswith('#')))
        for start,action,clip in segments(rows):
            images=[]
            for row in clip:
                path=source.parent/row['rgb_root_320']/row['six_views_320']
                image=imageio.imread(path)
                if image.shape!=(640,960,3):raise ValueError(f'Expected 640x960 RGB six-view mosaic: {path}, {image.shape}')
                images.append(image)
            for reverse in ([False] if a.no_reverse else [False,True]):
                label=INVERSE[action] if reverse else action
                rel=source.parent.relative_to(root)
                destination=out/rel/f"action_{start//20+1:03d}_f{start:06d}-f{start+20:06d}_{'reverse' if reverse else 'forward'}_srca{action}_labela{label}_{ACTIONS[label]}.mp4"
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():raise FileExistsError(destination)
                imageio.mimwrite(destination,images[::-1] if reverse else images,fps=21,codec='libx264',quality=8,macro_block_size=16)
                records.append(dict(video=Path(os.path.relpath(destination,Path.cwd())).as_posix(),prompt=ACTIONS[label].replace('_',' '),label_action_name=ACTIONS[label],video_frame_count=21,source_action=action,reverse=reverse))
    if not records:raise ValueError('No complete 20-step actions found')
    meta.parent.mkdir(parents=True,exist_ok=True)
    meta.write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(f'Wrote {len(records)} clips and {meta}; incomplete trailing actions omitted')
if __name__=='__main__':main()
