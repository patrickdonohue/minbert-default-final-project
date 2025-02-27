from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]
                if 'm' not in state.keys():
                    state['m'] = 0
                if 'v' not in state.keys():
                    state['v'] = 0
                if 't' not in state.keys():
                    state['t'] = 1
                # print('grad')
                # print(grad)
                # # print('group')
                # # print(group)
                # print('p')
                # print(p)
                # print(p.data)
                # print('state')
                # print(state)
                # print('self.param_groups')
                # print(self.param_groups)
                # print('group[params]')
                # print(group["params"])

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1 = group["betas"][0]
                beta2 = group["betas"][1]
                eps = group["eps"]
                wd = group["weight_decay"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                firstMoment = beta1 * state['m'] + (1- beta1) * grad
                secondMoment = beta2 * state['v'] + (1 - beta2) * torch.square(grad)

                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                alpha_t = alpha * math.sqrt((1 - beta2 ** state['t'])) / (1 - beta1 ** state['t'])
                # 3- Update parameters (p.data).
                update1 = alpha_t * firstMoment / (torch.sqrt(secondMoment) + eps) #alpha_t?
                #lambda is weight decay!
                update2 = p.data * wd * alpha
                #p.data -= alpha_t * firstMoment / (math.sqrt(secondMoment) + eps)
                temp = p.data
                p.data -= update1
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).
                # print('updates')
                # print(update1)
                # print(update2)
                p.data -= update2
                state['t'] += 1
                state['m'] = firstMoment
                state['v'] = secondMoment
                self.state[p] = state
                ### TODO
                #raise NotImplementedError


        return loss