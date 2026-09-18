import ast,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prepare=load('prepare','src/data/prepare.py');sft=load('sft_launch','src/sft/launch.py')
class ReleaseTests(unittest.TestCase):
 def test_python_syntax(self):
  for p in ROOT.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 def test_public_source_hygiene(self):
  import re
  for p in ROOT.rglob('*'):
   if not p.is_file() or '.git' in p.parts or p.suffix not in {'.py','.json','.yaml','.sh','.md','.toml'}:continue
   text=p.read_text()
   self.assertIsNone(re.search(r'[\u4e00-\u9fff]',text),str(p))
   self.assertIsNone(re.search(r'(?:/)(?:Users|home|mnt|ML-vePFS)/',text),str(p))
  ignore=(ROOT/'.gitignore').read_text().splitlines()
  self.assertIn('/data/',ignore);self.assertNotIn('data/',ignore)
 def test_shell_syntax(self):
  for p in (ROOT/'scripts').glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
 def test_relative_config_paths_and_pairs(self):
  for model in ['wan5b','wan1p3b']:
   configs=[json.loads((ROOT/'configs'/f'{model}_{s}.json').read_text()) for s in ['sft','stage2','stage3']]
   for c in configs:
    for k,v in c.items():
     if k in ['model_root','metadata','output_dir','cache_dir','teacher_checkpoint','generator_checkpoint']:
      self.assertFalse(Path(v).is_absolute());self.assertNotIn('..',Path(v).parts)
   a,b=configs[1:]
   for k in ['model_root','metadata','teacher_checkpoint','num_frames','height','width','cd_grid_steps','teacher_guidance','negative_prompt']:self.assertEqual(a[k],b[k],k)
   self.assertEqual(a['world_size'],8);self.assertEqual(a['max_steps'],3000);self.assertEqual(a['save_every'],1000)
   self.assertEqual(a['cd_grid_steps']%b['inference_steps'],0)
 def test_sft_command(self):
  for model in ['wan5b','wan1p3b']:
   cfg=json.loads((ROOT/'configs'/f'{model}_sft.json').read_text());cmd=sft.command(model,cfg)
   self.assertEqual(cmd[cmd.index('--dataset_base_path')+1],'.');self.assertEqual(cmd[cmd.index('--num_frames')+1],'21')
   self.assertEqual(cmd[cmd.index('--extra_inputs')+1],'input_image')
   self.assertNotIn('--height',cmd) # Preserve native video resolution.
   cfg['resume_from_checkpoint']='weights/resume.safetensors';self.assertIn('--resume_from_checkpoint',sft.command(model,cfg))
 def test_action_boundaries_and_reverse(self):
  rows=[dict(frame_id=i,action=-1 if i==0 else 10,action_name='start' if i==0 else 'move_up') for i in range(22)]
  clip=list(prepare.segments(rows));self.assertEqual(len(clip),1);self.assertEqual(len(clip[0][2]),21)
  self.assertEqual(prepare.INVERSE[10],11)
  for k,v in prepare.INVERSE.items():self.assertEqual(prepare.INVERSE[v],k)
  rows[10]['action']=8
  with self.assertRaises(ValueError):list(prepare.segments(rows))
 def test_airsim_roll_pitch_order(self):
  from types import SimpleNamespace
  # Execute only the setter, without importing AirSim or opening a connection.
  for name in ['collect_chain.py','collect_random.py']:
   tree=ast.parse((ROOT/'src/data'/name).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='set_vehicle_pose')
   node.returns=None
   for arg in node.args.args:arg.annotation=None
   calls=[]
   api=SimpleNamespace(Vector3r=lambda *x:x,to_quaternion=lambda *x:x,Pose=lambda pos,quat:(pos,quat))
   ns={'airsim':api};exec(compile(ast.Module(body=[node],type_ignores=[]),'setter','exec'),ns)
   client=SimpleNamespace(simSetVehiclePose=lambda *x:calls.append(x))
   ns['set_vehicle_pose'](client,[0,0,0],[.1,.2,.3])
   self.assertEqual(calls[0][0][1],(.2,.1,.3))
 def test_prepare_roundtrip(self):
  import numpy as np,imageio.v2 as imageio,csv
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);chain=root/'raw/ENV0/chain_0';images=chain/'rgb320/six_views';images.mkdir(parents=True)
   with (chain/'chain_0.csv').open('w') as f:
    f.write('# fixture\n');w=csv.DictWriter(f,fieldnames=['frame_id','action','action_name','rgb_root_320','six_views_320']);w.writeheader()
    for i in range(21):
     imageio.imwrite(images/f'{i:06d}.png',np.full((640,960,3),i*10,dtype=np.uint8))
     w.writerow(dict(frame_id=i,action=-1 if i==0 else 10,action_name='start' if i==0 else 'move_up',rgb_root_320='rgb320',six_views_320=f'six_views/{i:06d}.png'))
   subprocess.run([sys.executable,str(ROOT/'src/data/prepare.py'),'--raw-root','raw','--output','videos','--metadata','metadata.jsonl'],cwd=root,check=True)
   rows=[json.loads(x) for x in (root/'metadata.jsonl').read_text().splitlines()]
   self.assertEqual([r['prompt'] for r in rows],['move up','move down'])
   for row in rows:
    self.assertFalse(Path(row['video']).is_absolute());frames=imageio.mimread(root/row['video']);self.assertEqual(len(frames),21)
    self.assertEqual(frames[0].shape,(640,960,3))
    self.assertLess(abs(float(frames[0].mean())-(200 if row['reverse'] else 0)),4)
if __name__=='__main__':unittest.main()
