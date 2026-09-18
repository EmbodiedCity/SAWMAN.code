import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/distill/wan1p3b"))
import unittest
from unittest.mock import patch
import torch
import plain_cd as common
import distributed_train as train

class Tiny(torch.nn.Module):
    def __init__(self, value):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(value,dtype=torch.float32))
    def forward(self, x, t, context, clip, y, checkpointing=False):
        return x * self.weight + context
    def clip_grad_norm_(self, value):
        return torch.nn.utils.clip_grad_norm_(self.parameters(),value)

class Contracts(unittest.TestCase):
    def test_native_fun_conditioning_forward(self):
        model = train.NativeWan(torch.nn.Linear(1,1))
        args = {}
        def native(**kw):
            args.update(kw); return kw['latents']
        with patch.object(train,'model_fn_wan_video',native):
            model(torch.ones(1,16,6,2,2),torch.ones(1),torch.ones(1,2,3),
                  torch.ones(1,257,1280),torch.ones(1,20,6,2,2))
        self.assertFalse(args['fuse_vae_embedding_in_latents'])
        self.assertEqual(args['y'].shape[2],6)
        self.assertEqual(args['latents'].shape[2],6)
        self.assertIn('clip_feature',args)

    def test_mask_matches_native_pixel_to_latent_reshape(self):
        pixel = torch.ones(1,21,2,2); pixel[:,1:] = 0
        native = torch.cat([torch.repeat_interleave(pixel[:,:1],4,dim=1),pixel[:,1:]],dim=1)
        native = native.view(1,6,4,2,2).transpose(1,2)
        actual = torch.zeros(1,4,6,2,2); actual[:,:,:1] = 1
        self.assertTrue(torch.equal(native,actual))

    def test_exact_zero_boundary_and_guidance(self):
        model = Tiny(.2)
        x = torch.ones(1,16,6,2,2)
        case = dict(context=torch.tensor(2.),negative_context=torch.tensor(0.),clip=None,y=None)
        self.assertTrue(torch.equal(train.x0(model,x,0,case),x))
        pred = train.teacher_velocity(model,x,1,case,{'teacher_guidance':3.})
        self.assertTrue(torch.allclose(pred,torch.full_like(x,6.2)))
        fine = common.grid(48,5)
        self.assertTrue(torch.allclose(fine[::12],common.grid(4,5)))

    def test_small_fp32_master_update_survives(self):
        student,teacher,ema = Tiny(.2),Tiny(.25),Tiny(.2)
        opt = torch.optim.AdamW(student.parameters(),lr=2e-6)
        before = student.weight.detach().clone()
        case = dict(clean=torch.ones(1,16,6,2,2),context=torch.tensor(.1),negative_context=torch.tensor(0.),clip=None,y=None)
        cfg = dict(cd_grid_steps=48,sigma_shift=5,seed=0,teacher_guidance=3.,max_grad_norm=1.,ema_decay=.99)
        torch.manual_seed(0)
        metrics = train.stage2_step(student,teacher,ema,opt,case,cfg,1)
        self.assertNotEqual(float(student.weight),float(before))
        self.assertTrue(torch.isfinite(torch.tensor(metrics['cd_loss'])))
        self.assertEqual(opt.state[student.weight]['exp_avg'].dtype,torch.float32)

    def test_dmd_full_six_latents_and_both_optimizers(self):
        student, teacher, ema, fake = Tiny(.2), Tiny(.25), Tiny(.2), Tiny(.25)
        opt = torch.optim.AdamW(student.parameters(),lr=1e-6)
        critic = torch.optim.AdamW(fake.parameters(),lr=4e-7)
        case = dict(clean=torch.zeros(1,16,6,2,2),context=torch.tensor(.1),negative_context=torch.tensor(0.),clip=None,y=None)
        cfg = dict(seed=0,inference_steps=2,sigma_shift=5,teacher_guidance=3.,
                   critic_updates_per_generator=5,max_grad_norm=1.,ema_decay=.99)
        before = student.weight.detach().clone()
        torch.manual_seed(0)
        metrics = train.stage3_step(student,teacher,ema,fake,opt,critic,case,cfg,1)
        self.assertEqual(metrics['generator_updated'],1)
        self.assertGreater(metrics['dmd_loss'],0)
        self.assertNotEqual(float(before),float(student.weight))
        self.assertTrue(torch.isfinite(torch.tensor(metrics['fake_loss'])))
        frozen = student.weight.detach().clone()
        metrics = train.stage3_step(student,teacher,ema,fake,opt,critic,case,cfg,2)
        self.assertEqual(metrics['generator_updated'],0)
        self.assertTrue(torch.equal(frozen,student.weight))
        self.assertEqual(critic.state[fake.weight]['exp_avg'].dtype,torch.float32)
        torch.manual_seed(0)
        result = train.rollout(student,case,common.grid(2,5),1,False)
        self.assertEqual(result.shape,case['clean'].shape)
        self.assertGreater(float(result[:,:,:1].abs().sum()),0)

if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
