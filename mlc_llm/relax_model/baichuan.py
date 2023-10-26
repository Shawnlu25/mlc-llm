import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import tvm
from tvm import relax
from tvm.relax.testing import nn
from tvm import topi

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
    
class BaichuanAttention(nn.Module):
    def __init__(self, config: BaichuanConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length
        
        self.W_pack = nn.Parameter((3*self.hidden_size, self.hidden_size), dtype="float32", name="baichuan_attention_W_pack")
        self.o_proj = nn.Parameter((self.hidden_size, self.num_heads * self.head_dim), dtype="float32", name="baichuan_attention_o_proj")

    def forward(self, 
                hidden_states: relax.Expr,
                attention_mask: Optional[relax.Expr], 
                past_key_value: Optional[Tuple[relax.Expr]], 
                output_attentions: bool = False,
                use_cache: bool = False) -> Tuple[relax.Expr, Optional[relax.Expr], Optional[Tuple[relax.Expr]]]:
        bsz, q_len, _ = hidden_states.struct_info.shape

        proj = nn.emit(relax.op.linear(hidden_states, self.W_pack)) # [bsz, q_len, 3*hidden_size]
        
        qkv_states = nn.emit(
            relax.op.split(
                proj,
                indices_or_sections=3,
                axis=-1,
            )
        ) # [3, bsz, q_len, hidden_size]
        query_states = relax.TupleGetItem(qkv_states, 0) #[bsz, q_len, hidden_size]
        key_states = relax.TupleGetItem(qkv_states, 1)
        value_states = relax.TupleGetItem(qkv_states, 2)
        
        query_states = nn.emit(relax.op.permute_dims(relax.op.reshape(query_states, (bsz, q_len, self.num_heads, self.head_dim)), [0,2,1,3])) # [bsz, num_query_heads, q_len, head_dim]
        key_states = nn.emit(relax.op.permute_dims(relax.op.reshape(key_states, (bsz, q_len, self.num_heads, self.head_dim)), [0,2,1,3]))
        value_states = nn.emit(relax.op.permute_dims(relax.op.reshape(value_states, (bsz, q_len, self.num_heads, self.head_dim)), [0,2,1,3]))

        kv_seq_len = key_states.struct_info.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].struct_info.shape[-2]
            key_states = nn.emit(relax.op.concat([past_key_value[0], key_states], axis=-2))
            value_states = nn.emit(relax.op.concat([past_key_value[1], value_states], axis=-2))

        if use_cache:
            past_key_value = nn.emit(relax.Tuple([key_states, value_states]))
        else:
            past_key_value = None

        attn_weights = nn.emit(
            relax.op.matmul(
                query_states,
                relax.op.permute_dims(key_states, [0,1,3,2]),
            ) / relax.const(math.sqrt(self.head_dim), dtype="float32")
        )
        if attention_mask is not None:
            if q_len == 1:
                if len(attention_mask.struct_info.shape) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = nn.emit(attn_weights + attention_mask)
            attn_weights = nn.emit(
                relax.op.maximum(
                    attn_weights,
                    relax.const(
                        tvm.tir.min_value(attn_weights.struct_info.dtype).value,
                        attn_weights.struct_info.dtype,
                    ),
                )
            )
            
        attn_weights = nn.emit(relax.op.nn.softmax(attn_weights, axis=-1))
        attn_output = nn.emit(relax.op.matmul(attn_weights, value_states))
        attn_output = nn.emit(relax.op.permute_dims(attn_output, [0,2,1,3]))
        attn_output = nn.emit(relax.op.reshape(attn_output, (bsz, q_len, self.hidden_size)))
        attn_output = nn.emit(relax.op.linear(attn_output, self.o_proj))

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class BaichuanLayer(nn.Module):
    def __init__(self, config: BaichuanConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = BaichuanAttention(config)
        self.mlp = BaichuanMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = BaichuanRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = BaichuanRMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(self, 
                hidden_states: relax.Expr,
                attention_mask: Optional[relax.Expr] = None,
                past_key_value: Optional[Tuple[relax.Expr]] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                ) -> Tuple[relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states, attention_mask, past_key_value, output_attentions, use_cache)
        hidden_states = nn.emit(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = nn.emit(residual + self.mlp(hidden_states))

        outputs = (hidden_states, )
        if use_cache:
            outputs += (present_key_value, )
        return outputs



def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def _gen_alibi_mask(tensor, n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask

class BaichuanAttentionOriginal(torch.nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = torch.nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=False
        )
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = (
            proj.unflatten(-1, (3, self.hidden_size))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )
        query_states = (
            proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def build_relax(nn_module) -> tvm.ir.IRModule:
    # relax.BlockBuilder can construct end-to-end models step by step in an IRModule that starts empty
    bb = relax.BlockBuilder()
    # the relax nn module
    config = BaichuanConfig()
    model = nn_module(config)
    # create a function called `linear` in the IRModule
    with bb.function("main"):
        # define input placeholder to the relax nn module
        hidden_states = nn.Placeholder((2, 1024, 5120), dtype="float32", name="hidden_states")
        past_key = nn.Placeholder((2, 40, 64, 128), dtype="float32", name="past_key")
        past_value = nn.Placeholder((2, 40, 64, 128), dtype="float32", name="past_value")
        attention_mask = nn.Placeholder((40, 1024, 1088), dtype="float32", name="attention_mask")

        # build dataflow block
        with bb.dataflow():
            # call forward function of relax nn module to build IRModule
            output = model(hidden_states, attention_mask, (past_key, past_value), use_cache=True, output_attentions=True)
            # The params of the constructed IRModule
            params = [hidden_states, attention_mask, past_key, past_value] + model.parameters()
            # return value of the dataflow block
            output = [o for o in output if o is not None]
            gv = bb.emit_output(output)
        # return value and params of the Relax function `linear`
        bb.emit_func_output(gv, params)
    return bb.get()



mod_th = BaichuanAttentionOriginal(BaichuanConfig())
hidden_states_th = torch.randn(2, 1024, 5120)
alibi = _gen_alibi_mask(hidden_states_th, 40, 1088)
attention_mask = alibi[:, :1024, :1088]
print("attention_mask.shape", attention_mask.shape)

past_key_th = torch.randn(2, 40, 64, 128)
past_value_th = torch.randn(2, 40, 64, 128)
attn_output, attn_weight, key_value =  mod_th(hidden_states_th, attention_mask, (past_key_th, past_value_th), use_cache=True, output_attentions=True)
print("attn_output.shape", attn_output.shape)
print("attn_weight.shape", attn_weight.shape)
print("past_key.shape", key_value[0].shape)
print("past_value.shape", key_value[1].shape)


mod = build_relax(BaichuanAttention)
mod = relax.transform.LegalizeOps()(mod)
mod.show()

with tvm.target.Target("cuda"):
    mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

ex = relax.build(mod, target="cuda")
vm = relax.VirtualMachine(ex, tvm.cuda())

hidden_states_nd = tvm.nd.array(hidden_states_th.numpy(), device=tvm.cuda())
attention_mask_nd = tvm.nd.array(attention_mask.numpy(), device=tvm.cuda())
past_key_nd = tvm.nd.array(past_key_th.numpy(), device=tvm.cuda())
past_value_nd = tvm.nd.array(past_value_th.numpy(), device=tvm.cuda())
w_pack_nd = tvm.nd.array(mod_th.W_pack.weight.detach().numpy(), device=tvm.cuda())
o_proj_nd = tvm.nd.array(mod_th.o_proj.weight.detach().numpy(), device=tvm.cuda())


a=vm["main"](hidden_states_nd, attention_mask_nd, past_key_nd, past_value_nd, w_pack_nd, o_proj_nd)
print(a[0].numpy().shape)
print(a[1].numpy().shape)
print(a[2][0].numpy().shape)
print(a[2][1].numpy().shape)


