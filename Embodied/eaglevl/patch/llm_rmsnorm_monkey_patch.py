import transformers


def replace_llm_rmsnorm_with_fused_rmsnorm():
    try:
        from functools import partial

        from apex.normalization import FusedRMSNorm
        LlamaRMSNorm = partial(FusedRMSNorm, eps=1e-6)   # noqa
        transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
        transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm = LlamaRMSNorm
        print('Discovered apex.normalization.FusedRMSNorm - will use it instead of LlamaRMSNorm, currently used by: LLama and Qwen2')
    except ImportError:
        # using the normal LlamaRMSNorm
        pass
    except Exception:
        print('discovered apex but it failed to load, falling back to LlamaRMSNorm')
        pass
