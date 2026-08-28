import numpy as np
import torch
import os

IGNORE_INDEX = -100

import torch.distributed as dist    
import traceback


def pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        if not feat['attention_mask'].dtype == torch.long:
            feat['attention_mask'] = feat['input_ids'].ne(pad_id)
        # pad position_ids if present
        if 'position_ids' in feat:
            pos = feat['position_ids']
            temp_position_ids = torch.LongTensor([1] * max_item_length)
            temp_position_ids[:pos.shape[0]] = pos
            feat['position_ids'] = temp_position_ids

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    if 'sub_sample_lengths' in first:
        batch['sub_sample_lengths'] = [f['sub_sample_lengths'] for f in features]
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'sub_sample_lengths') and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


def concat_pad_data_collator(features, pad_id=0):
    # print(features[0].keys())
    # print(type(features))
    # print(features[0].keys())

    first = features[0]    
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        if not feat['attention_mask'].dtype == torch.long:
            feat['attention_mask'] = feat['input_ids'].ne(pad_id)
        # pad position_ids if present
        if 'position_ids' in feat:
            pos = feat['position_ids']
            temp_position_ids = torch.LongTensor([1] * max_item_length)
            temp_position_ids[:pos.shape[0]] = pos
            feat['position_ids'] = temp_position_ids

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    if 'sub_sample_lengths' in first:
        batch['sub_sample_lengths'] = [f['sub_sample_lengths'] for f in features]

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags', 'sub_sample_lengths') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                try:
                    batch[k] = torch.stack([f[k] for f in features])
                except:
                    print(batch['sub_sample_lengths'])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])

        if k in ('sub_sample_lengths'):
            if isinstance(v, torch.Tensor): 
                batch[k] = [f[k] for f in features]
            elif isinstance(v, np.ndarray):
                batch[k] =np.stack([f[k] for f in features])
            else: # for sub_sample_lengths
                batch[k] = [f[k] for f in features]
    return batch


def pad_data_collator_for_anyres(features, pad_id=0):
    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        orig_len = feat['input_ids'].shape[0]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:orig_len] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        # pad attention_mask: 1 for real tokens, 0 for padding
        temp_attention_mask = torch.zeros(max_item_length, dtype=torch.long)
        temp_attention_mask[:orig_len] = 1
        feat['attention_mask'] = temp_attention_mask
        # pad position_ids if present
        if 'position_ids' in feat:
            pos = feat['position_ids']
            temp_position_ids = torch.LongTensor([1] * max_item_length)
            temp_position_ids[:pos.shape[0]] = pos
            feat['position_ids'] = temp_position_ids
        # pad loss_weight if present: 0 for padding positions
        if 'loss_weight' in feat:
            lw = feat['loss_weight']
            # Ensure no size mismatches: take min of max_item_length and lw.shape[0]
            actual_len = min(max_item_length, lw.shape[0])
            temp_loss_weight = torch.zeros(max_item_length, dtype=torch.float32)
            temp_loss_weight[:actual_len] = lw[:actual_len]
            feat['loss_weight'] = temp_loss_weight

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    if 'sub_sample_lengths' in first:
        batch['sub_sample_lengths'] = [f['sub_sample_lengths'] for f in features]
    
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'sub_sample_lengths', 'pixel_values', 'image_flags', 'image_grid_hws') and v is not None and not isinstance(v, str):
            try:
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
            except:
                print(k)
                for f in features:
                    print(f[k].shape)
                traceback.print_exc()
                exit()

        if k == 'image_flags':
            batch[k] = torch.stack([f[k] for f in features], dim=0)
    
        if k == 'pixel_values':
            batch[k] = torch.cat([f[k] for f in features], dim=0)
    
        if k == 'image_grid_hws':
            batch[k] = torch.cat([torch.from_numpy(f[k]) for f in features], dim=0)

    return batch


def get_collator(collator_type='default'):
    if collator_type == 'default':
        return concat_pad_data_collator
    elif collator_type == 'for_anyres':
        return pad_data_collator_for_anyres