import math
import os
from dataclasses import dataclass

import torch
import tvm
from tvm import relax
from tvm.relax.testing import nn

@dataclass
class BaichuanConfig:
    vocab_size: int = 64000
    hidden_size: int = 5120
    intermediate_size: int = 13696
    num_hidden_layers: int = 40
    num_attention_heads: int = 40
    hidden_act: str = "silu"
    model_max_length: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    gradient_checkpointing: bool = False
    z_loss_weight: int = 0


class BaichuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, epsilon=1e-6):
        self.weight = nn.Parameter((hidden_size, ), dtype="float32", name="baichuan_rmsnorm_weight")
        self.epsilon = tvm.tir.const(epsilon, dtype="float32", name="baichuan_rmsnorm_eps")
    
    def forward(self, hidden_states: relax.Expr) -> relax.Var:
        if hidden_states.struct_info.dtype != "float32":
            hidden_states = nn.emit(relax.op.astype(hidden_states, "float32"))
        return nn.emit(relax.op.nn.rms_norm(hidden_states, self.weight, self.epsilon))

class BaichuanMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_proj = nn.Parameter((intermediate_size, hidden_size), dtype="float32", name="baichuan_mlp_gate_proj")
        self.down_proj = nn.Parameter((hidden_size, intermediate_size), dtype="float32", name="baichuan_mlp_down_proj")
        self.up_proj = nn.Parameter((intermediate_size, hidden_size), dtype="float32", name="baichuan_mlp_up_proj")
    
    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(relax.op.nn.silu(relax.op.linear(x, self.gate_proj)) * relax.op.linear(x, self.up_proj), self.down_proj))
    




def build_relax(nn_module) -> tvm.ir.IRModule:
    # relax.BlockBuilder can construct end-to-end models step by step in an IRModule that starts empty
    bb = relax.BlockBuilder()
    # the relax nn module
    model = nn_module(784, 128)
    # create a function called `linear` in the IRModule
    with bb.function("main"):
        # define input placeholder to the relax nn module
        input = nn.Placeholder((1, 784), dtype="float32", name="input")
        # build dataflow block
        with bb.dataflow():
            # call forward function of relax nn module to build IRModule
            logits = model(input)
            # The params of the constructed IRModule
            params = [input] + model.parameters()
            # return value of the dataflow block
            gv = bb.emit_output(logits)
        # return value and params of the Relax function `linear`
        bb.emit_func_output(gv, params)
    return bb.get()
