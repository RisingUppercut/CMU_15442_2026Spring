import numpy as np

import sys
sys.path.append("..")

from typing import List, Tuple, Dict
from model.Layers import ReLULayer, CrossEntropyLossLayer
from mpi4py import MPI

class ZeroDPStage3FCLayer(object):
    def __init__(
        self,
        comm,
        in_dim: int,
        out_dim: int,
        dp_size: int,
        full_w: np.ndarray = None,
        full_b: np.ndarray = None,
        seed: int = 15442,
    ):
        """A standalone FC layer for ZeRO-DP Stage 3 assignment tasks.

        Parameters
        ----------
            comm : Communicator
                data parallel communicator with ``Get_size()==dp_size``

            in_dim : int
                input feature dimension

            out_dim : int
                output feature dimension

            dp_size : int
                data parallel size

            full_w : np.ndarray, optional
                full weight matrix of shape (in_dim, out_dim) for testing correctness, 
                defaults to None (will be initialized internally)
                Note: We don't guarantee the number of elements in full_w is divisible 
                by dp_size. You should handle the padding logic in partitioning.
            
            full_b : np.ndarray, optional
                full bias vector of shape (1, out_dim) for testing correctness, 
                defaults to None (will be initialized internally)
                Note: We don't guarantee the number of elements in full_b is divisible 
                by dp_size. You should handle the padding logic in partitioning.

            seed : int, optional
                seed for deterministic parameter initialization
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.dp_size = dp_size
        self.in_dim = in_dim
        self.out_dim = out_dim

        assert self.dp_size > 0
        assert self.comm.Get_size() == self.dp_size

        self.name = f"zero-dp-stage3-fc-{in_dim}x{out_dim}"
        self.x = None

        rng = np.random.default_rng(seed)
        if full_w is None:
            full_w = rng.standard_normal((self.in_dim, self.out_dim)).astype(np.float64)
            full_w = full_w / np.sqrt(self.in_dim)
        if full_b is None:
            full_b = rng.standard_normal((1, self.out_dim)).astype(np.float64)
            full_b = full_b / np.sqrt(self.out_dim)

        self.w_numel = self.in_dim * self.out_dim
        self.b_numel = self.out_dim

        self.w_shard, self.w_shard_size = self._partition_flat_tensor(
            full_w, self.rank, self.dp_size
        )
        self.b_shard, self.b_shard_size = self._partition_flat_tensor(
            full_b, self.rank, self.dp_size
        )

        self.grad_w_shard = np.zeros_like(self.w_shard)
        self.grad_b_shard = np.zeros_like(self.b_shard)


    def _partition_flat_tensor(self, tensor: np.ndarray, shard_idx: int, num_shards: int) -> Tuple[np.ndarray, int]:
        """Partition a tensor by flattening first.

        Parameters
        ----------
            tensor : np.ndarray
                input tensor of arbitrary shape.

            shard_idx : int
                local shard index in [0, num_shards).

            num_shards : int
                number of equal shards. In our case, num_shards == dp_size.

        Returns
        -------
            tensor_shard : np.ndarray
                the local shard of the input tensor for this rank, flattened to 1D.
                This should include the padded elements if there is any.
            
            shard_size : int
                the number of elements in each shard **including padding**.
        """

        """TODO: Your code here"""

        # Hint: You need to handle the case when tensor.numel() is not divisible by 
        # num_shards by padding zeros at the end of the flattened tensor before 
        # partitioning. The returned shard should INCLUDE the padded elements.
        # We keep track of the original shard_size (without padding) for

        shard_size = (tensor.size + num_shards - 1) // num_shards
        padding_size = shard_size * num_shards - tensor.size
        flat_tensor = tensor.flatten()

        if padding_size > 0:
            flat_tensor = np.pad(flat_tensor, (0, padding_size), mode='constant', constant_values=0.)
        
        start = shard_size * shard_idx 
        end = shard_size * (shard_idx + 1)

        return flat_tensor[start:end], shard_size
        # return (np.empty(8), 8)

    def zero_grad(self):
        self.grad_w_shard = np.zeros_like(self.w_shard)
        self.grad_b_shard = np.zeros_like(self.b_shard)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass under ZeRO-DP Stage 3.

        Parameters
        ----------
            x : np.ndarray
                local input activation of shape ``(batch_size, in_dim)``

        Returns
        -------
            out : np.ndarray
                local output activation of shape ``(batch_size, out_dim)``

        Task
        ----
            Implement forward with Stage-3 parameter gathering:
            1) All-gather weights.
            2) Compute FC output.
        """
        self.x = x

        """TODO: Your code here"""
        collected_w = np.empty((self.w_shard_size * self.dp_size), dtype=self.w_shard.dtype)
        self.comm.Allgather(self.w_shard, collected_w)
        w = collected_w[:self.in_dim*self.out_dim].reshape(self.in_dim, self.out_dim)

        collected_b = np.empty((self.b_shard_size * self.dp_size), dtype=self.w_shard.dtype)
        self.comm.Allgather(self.b_shard, collected_b)
        b = collected_b[:self.out_dim].reshape(1, self.out_dim)

        out = self.x @ w + b
        return out 
        # raise NotImplementedError

    def backward(self, output_grad: np.ndarray) -> List[np.ndarray]:
        """Backward pass under ZeRO-DP Stage 3.

        Parameters
        ----------
            output_grad : np.ndarray
                local output gradient of shape ``(batch_size, out_dim)``

        Returns
        -------
            grad_x : List[np.ndarray]
                local input gradient of shape ``(batch_size, in_dim)``,
                wrapped in a single-element list for compatibility.

        Side Effects
        ------------
            should update the following attributes in-place:
            - ``self.grad_w_shard``
            - ``self.grad_b_shard``

        Task
        ----
            Implement backward with Stage-3 communication:
            1) Reconstruct full parameters.
            2) Compute local full gradients.
            3) Reduce-scatter flattened full gradients into local shards.
            4) Compute and return ``grad_x``.
            Note: Make sure to update `self.grad_w_shard` and `self.grad_b_shard` in-place.
            The autograder will call this method and expect the gradients to be populated in 
            these attributes for the optimizer step.
        """

        """TODO: Your code here"""

        collected_w = np.empty((self.w_shard_size * self.dp_size), dtype=self.w_shard.dtype)
        self.comm.Allgather(self.w_shard, collected_w)
        w = collected_w[:self.in_dim*self.out_dim].reshape(self.in_dim, self.out_dim)

        # collected_b = np.empty((self.b_shard_size * self.dp_size), dtype=self.w_shard.dtype)
        # self.comm.Allgather(self.b_shard, collected_b)
        # b = collected_b[:self.out_dim].reshape(1, self.out_dim)

        
        full_grad_w = self.x.T @ output_grad 
        full_grad_b = np.sum(output_grad, axis=0)
        
        w_shard_size = (full_grad_w.size + self.dp_size - 1) // self.dp_size
        w_padding_size = w_shard_size * self.dp_size - full_grad_w.size
        flat_grad_w = full_grad_w.flatten()

        if w_padding_size > 0:
            flat_grad_w = np.pad(flat_grad_w, (0, w_padding_size), mode='constant', constant_values=0.)

        b_shard_size = (full_grad_b.size + self.dp_size - 1) // self.dp_size
        b_padding_size = b_shard_size * self.dp_size - full_grad_b.size
        flat_grad_b = full_grad_b.flatten()

        if b_padding_size > 0:
            flat_grad_b = np.pad(flat_grad_b, (0, b_padding_size), mode='constant', constant_values=0.)

        grad_w_shard = np.empty(w_shard_size, dtype=flat_grad_w.dtype)
        grad_b_shard = np.empty(b_shard_size, dtype=flat_grad_b.dtype)

        self.comm.Reduce_scatter(flat_grad_w, grad_w_shard, op=MPI.SUM)
        self.comm.Reduce_scatter(flat_grad_b, grad_b_shard, op=MPI.SUM)

        self.grad_w_shard[:] = grad_w_shard
        self.grad_b_shard[:] = grad_b_shard
        
        grad_x = output_grad @ w.T

        return [grad_x]
        # raise NotImplementedError


class ZeroDPMLPModel(object):
    def __init__(self, comm, dp_size, feature_dim, hidden_dim, output_dim):
        self.layers = [
            ZeroDPStage3FCLayer(
                comm=comm,
                in_dim=feature_dim,
                out_dim=hidden_dim,
                dp_size=dp_size,
            ),
            ReLULayer(),
            ZeroDPStage3FCLayer(
                comm=comm,
                in_dim=hidden_dim,
                out_dim=output_dim,
                dp_size=dp_size,
            ),
        ]
        self.loss_layer = CrossEntropyLossLayer()
    
    def forward(self, x, y):
        """

        :param x: input images of shape (batch_size, feature_dim)
        :param y: labels of shape (batch_size, )
        :return: loss
        """
        y_one_hot = np.zeros((x.shape[0], 10))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        x = self.layers[0].forward(x)
        x = self.layers[1].forward(x)
        x = self.layers[2].forward(x)
        predict = np.argmax(x, axis=1)
        acc = (y == predict).sum() / y.shape[0]
        loss = self.loss_layer.forward(x, y_one_hot)
        return loss, acc
    
    def backward(self):
        """Backward pass to populate gradients in each layer."""
        grad = self.loss_layer.backward()[0]
        grad = self.layers[2].backward(grad)[0]
        grad = self.layers[1].backward(grad)[0]
        _ = self.layers[0].backward(grad)
    
    def zero_grad(self):
        self.layers[0].zero_grad()
        self.layers[2].zero_grad()


class ZeroDPAdam(object):
    def __init__(
        self,
        layers,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """A minimal Adam optimizer for sharded parameters in ZeRO-DP Stage 3.

        Parameters
        ----------
            layers : list[ZeroDPStage3FCLayer]
                list of stage-3 FC layers

            lr, beta1, beta2, eps : float
                Adam hyperparameters

        Notes
        -----
            Optimizer states are maintained per local shard only.
            State dict key: ``(layer_idx, param_name)`` where param_name in {"w", "b"}.
        """
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.step_idx = 0
        # State dict format: { (layer_idx, param_name): {"m": np.ndarray, "v": np.ndarray} }
        self.state : Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}

    def _iter_sharded_params(self):
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, ZeroDPStage3FCLayer):
                yield (layer_idx, "w"), layer.w_shard, layer.grad_w_shard
                yield (layer_idx, "b"), layer.b_shard, layer.grad_b_shard
            elif isinstance(layer, ReLULayer):
                continue
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

    def step(self) -> None:
        """Run one Adam update on all local parameter shards.

        Task
        ----
            Implement ZeRO-style optimizer state management:
            1) Iterate over all layers and parameter shards to get local params and grads.
            2) Update parameters and states in-place using Adam update rule.

        """
        self.step_idx += 1

        # Hints:
        # - Use self.state[(layer_idx, "w")] and self.state[(layer_idx, "b")] to store moments.
        # - Carefully think about the shapes of parameters, gradients, and states.
        # - Adam update:
        #   m = beta1*m + (1-beta1)*grad
        #   v = beta2*v + (1-beta2)*(grad*grad)
        #   m_hat = m / (1-beta1^t)
        #   v_hat = v / (1-beta2^t)
        #   p -= lr * m_hat / (sqrt(v_hat) + eps)
        # - If you don't understand the Adam update rule, you could consult your AI assistant or
        #  refer to the original Adam paper: https://arxiv.org/abs/1412.6980

        # Think of these questions:
        # 1. What's the difference between single-process Adam and ZeRO-Adam 
        # in terms of optimizer state management?
        # 2. Do we need to gather the full optimizer state across data parallel 
        # ranks for the update? Why or why not?
        # 3. If we need to do any communication, when and what to communicate?
        # If not, why?

        for key, param, grad in self._iter_sharded_params():
            if key not in self.state:
                self.state[key] = {
                    "m": np.zeros_like(param),
                    "v": np.zeros_like(param),
                }

            """TODO: Your code here"""
            state = self.state[key]
            m, v = state["m"], state["v"]
            
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            
            m_hat = m / (1 - self.beta1 ** self.step_idx)
            v_hat = v / (1 - self.beta2 ** self.step_idx)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # raise NotImplementedError