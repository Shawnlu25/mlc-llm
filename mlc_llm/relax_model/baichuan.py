import math
import os
from dataclasses import dataclass

import torch
import tvm
from tvm import relax
from tvm.relax.testing import nn

@dataclass
class BaichuanConfig:
    dtype: str = "float16"
    vocab_size: int = 125696
    word_embed: int = 4096
    rms_norm_eps: float = 1e-6

MODEL_CONFIG = {
    "Baichuan2-7B-Chat": {}
}

class BaichuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, epsilon=1e-6):
        super().__init__()

        # TODO: Is it necessary to name it?
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
