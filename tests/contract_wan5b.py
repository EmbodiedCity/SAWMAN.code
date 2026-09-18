import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/distill/wan5b"))
import unittest
from unittest.mock import patch
import torch
import distributed_train as training


class DMDContractTests(unittest.TestCase):
    def test_identical_scores_have_zero_dmd_gradient(self):
        generated = torch.randn(1, 48, 6, 2, 2)
        score = torch.randn_like(generated)
        gradient = training.dmd_gradient(generated, score, score)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_gradient_descent_moves_generator_toward_real_prediction(self):
        generated = torch.zeros(1, 48, 6, 2, 2, requires_grad=True)
        real = torch.ones_like(generated)
        fake = torch.zeros_like(generated)
        gradient = training.dmd_gradient(generated.detach(), real, fake)
        target = generated[:, :, 1:].detach() - gradient
        loss = .5 * (generated[:, :, 1:] - target).square().mean()
        loss.backward()
        self.assertTrue((generated.grad[:, :, 1:] < 0).all())
        self.assertEqual(generated.grad[:, :, :1].count_nonzero(), 0)

    def test_rollout_cannot_read_future_ground_truth(self):
        first = torch.randn(1, 48, 1, 2, 2)
        case = {"first": first, "context": torch.zeros(1, 1, 1), "clean": torch.zeros(1, 48, 6, 2, 2)}
        def prediction(model, x, sigma, case, checkpointing=False):
            return training.common.clamp_first(x * .9, case["first"])
        sigmas = training.common.grid(4, 5)
        with patch.object(training, "x0", side_effect=prediction):
            torch.manual_seed(42)
            a = training.rollout(None, case, sigmas, 3, False)
            case["clean"].fill_(999.)
            torch.manual_seed(42)
            b = training.rollout(None, case, sigmas, 3, False)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[:, :, :1], first))


if __name__ == "__main__":
    unittest.main()
