from .fused_ops.fused_rms_norm import LigerRMSNorm
from .fused_ops.fused_rotary_pos_emb import liger_rotary_pos_emb
from .fused_ops.fused_swiglu import LigerSwiGLUMLP
try:
    import torch
    from torch import nn

    from flash_attn.ops.fused_dense import fused_mlp_func

    class FusedSiglipMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            # assert activation in ["gelu_approx", "relu", "sqrelu"]
            in_features = config.hidden_size
            out_features = config.hidden_size
            hidden_features = config.intermediate_size
            self.activation = 'gelu_approx'
            self.return_residual = False
            self.checkpoint_lvl = 2
            self.heuristic = "auto"
            self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
            self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

        def forward(self, x, process_group=None):
            dtype = x.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype()
            if self.heuristic == "auto":
                if self.activation == "gelu_approx":
                    if torch.cuda.get_device_capability("cuda") == (9, 0):
                        heuristic = -1
                    else:
                        cuda_ver = tuple(map(int, torch.version.cuda.split(".")))
                        heuristic = 0 if cuda_ver >= (11, 8) else (1 if dtype == torch.float16 else -1)
                else:
                    heuristic = 0
            else:
                heuristic = self.heuristic
            out = fused_mlp_func(
                x,
                self.fc1.weight,
                self.fc2.weight,
                self.fc1.bias,
                self.fc2.bias,
                activation=self.activation,
                save_pre_act=self.training,
                return_residual=self.return_residual,
                checkpoint_lvl=self.checkpoint_lvl,
                heuristic=heuristic,
                process_group=process_group,
            )   
            return out
except:
    FusedSiglipMLP = None

def replace_siglip_fused_ops():
    from eaglevl.model import siglip
    # print("replace siglip fused ops")
    if FusedSiglipMLP is not None:
        print("replace siglip mlp")
        siglip.SiglipMLP = FusedSiglipMLP
    else:
        print("You are trying to use fused siglip mlp, but it is not available")

def replace_liger_fused_ops():
    # from eaglevl.model.qwen2 import modeling_qwen2 as custom_modeling_qwen2
    # print("replace liger fused ops")
    # custom_modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP
    # custom_modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm
    # modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    from transformers.models.qwen2 import modeling_qwen2
    modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP
    modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm
    # modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb # TODO check fp32
    
    from transformers.models.llama import modeling_llama
    modeling_llama.LlamaMLP = LigerSwiGLUMLP
    modeling_llama.LlamaRMSNorm = LigerRMSNorm

    from transformers.models.qwen3 import modeling_qwen3
    modeling_qwen3.Qwen3MLP = LigerSwiGLUMLP
    modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm
    modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
    
    



