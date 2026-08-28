# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.distributed as dist

from .comm import (all_to_all, gather_forward_split_backward,
                   split_forward_gather_backward)
from .globals import get_pg_manager


def pre_process_for_sequence_parallel_attn(query_states,
                                           key_states,
                                           value_states,
                                           scatter_dim=2,
                                           gather_dim=1):
    b, s_div_sp, h, d = query_states.shape
    sp = get_pg_manager().ulysses_sequence_parallel_world_size


    # (b, s_div_sp, insp*h, d/insp) -> (b, s, insp*h/sp, d/insp)
    sequence_parallel_group = get_pg_manager().ulysses_sequence_parallel_group
    query_states = all_to_all(
        query_states,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)
    key_states = all_to_all(
        key_states,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)
    value_states = all_to_all(
        value_states,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)


    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(attn_output,
                                            scatter_dim=1,
                                            gather_dim=2):
    sp = get_pg_manager().ulysses_sequence_parallel_world_size
    # insp = get_inner_sequence_parallel_world_size()
    b, s, h_mul_insp_div_sp, d = attn_output.shape
    h = h_mul_insp_div_sp * sp
    s_div_sp = s // sp


    # (b, s, insp*h/sp, d/insp) -> (b, s_div_sp, insp*h, d/insp)
    sequence_parallel_group = get_pg_manager().ulysses_sequence_parallel_group
    output = all_to_all(
        attn_output,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)

    return output

