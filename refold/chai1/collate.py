# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import dataclasses
import logging
from typing import Any

import torch
import torch.nn.functional as F

from chai_lab.data.collate.utils import get_pad_sizes
from chai_lab.data.dataset.all_atom_feature_context import AllAtomFeatureContext
from chai_lab.data.features.feature_factory import FeatureFactory
from chai_lab.model.utils import (
    get_block_atom_pair_mask,
    get_qkv_indices_for_blocks,
)
from chai_lab.utils.dict import list_dict_to_dict_list

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Collate:
    feature_factory: FeatureFactory
    num_query_atoms: int
    num_key_atoms: int

    def __call__(
        self,
        feature_contexts: list[AllAtomFeatureContext],
    ) -> dict[str, Any]:
        raw_batch = self._collate(feature_contexts)
        prepared_batch = self._post_collate(raw_batch)
        return prepared_batch

    def _collate(
        self,
        feature_contexts: list[AllAtomFeatureContext],
    ) -> dict[str, Any]:
        # Get the pad sizes, finding the max number of tokens/atoms/bonds in the batch.
        pad_sizes = get_pad_sizes([p.structure_context for p in feature_contexts])

        # Pad each feature context to the max sizes
        padded_feature_contexts = [
            feature_context.pad(
                n_tokens=pad_sizes.n_tokens,
                n_atoms=pad_sizes.n_atoms,
            )
            for feature_context in feature_contexts
        ]

        # Convert all the input data into dicts, for each feature context
        inputs_per_context = [e.to_dict() for e in padded_feature_contexts]

        # Stack the dict inputs into a single batch dict, across all feature contexts
        # Handle tensors with different feature dimensions by padding to max dimension
        def stack_with_padding(v):
            if not isinstance(v[0], torch.Tensor):
                return v
            
            # Check if all tensors have the same shape
            first_shape = v[0].shape
            all_same_shape = all(t.shape == first_shape for t in v)
            
            if all_same_shape:
                # All tensors have the same shape, can stack directly
                return torch.stack(v, dim=0)
            
            # Tensors have different shapes, need to pad to max dimensions
            # Log warning about shape mismatch
            shapes = [t.shape for t in v]
            logger.warning(
                f"Tensors have different shapes in batch: {shapes}. "
                f"Padding to max dimensions: {max(shapes, key=lambda s: tuple(s))}"
            )
            
            # Find max dimensions for each axis
            max_dims = list(first_shape)
            for t in v[1:]:
                for i, dim_size in enumerate(t.shape):
                    if i < len(max_dims):
                        max_dims[i] = max(max_dims[i], dim_size)
                    else:
                        max_dims.append(dim_size)
            
            # Pad all tensors to max dimensions
            padded_tensors = []
            for idx, t in enumerate(v):
                # Calculate padding for each dimension (F.pad expects padding from last dim to first)
                padding = []
                for i in range(len(max_dims) - 1, -1, -1):  # Reverse order for F.pad
                    if i < len(t.shape):
                        pad_size = max_dims[i] - t.shape[i]
                        padding.extend([0, pad_size])
                    else:
                        # Tensor has fewer dimensions, pad entire dimension
                        padding.extend([0, max_dims[i]])
                
                if any(p > 0 for p in padding):
                    padded_t = F.pad(t, padding, mode='constant', value=0)
                    logger.debug(
                        f"Padded tensor {idx} from shape {t.shape} to {padded_t.shape} "
                        f"(max_dims={max_dims}, padding={padding})"
                    )
                else:
                    padded_t = t
                padded_tensors.append(padded_t)
            
            return torch.stack(padded_tensors, dim=0)
        
        batched_inputs = {
            k: stack_with_padding(v)
            for k, v in list_dict_to_dict_list(inputs_per_context).items()
        }

        # Make a batch dict
        batch = dict(inputs=batched_inputs)
        batch['fasta_path'] = [feature_context.fasta_path for feature_context in feature_contexts]
        return batch

    def _post_collate(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        """
        takes a list of processed multi-chain systems,
        returns a dictionary with batched tensors to feed in the model forward method
        and any other necessary data for the task/losses
        """
        raw_b_i = raw_batch["inputs"]

        # prepare atom pair block data:
        atom_exists_mask = raw_b_i["atom_exists_mask"]
        block_q_atom_idces, block_kv_atom_idces, kv_mask = get_qkv_indices_for_blocks(
            atom_exists_mask.shape[1],
            self.num_query_atoms,
            self.num_key_atoms,
            atom_exists_mask.device,
        )
        block_atom_pair_mask = get_block_atom_pair_mask(
            atom_single_mask=raw_b_i["atom_ref_mask"],
            q_idx=block_q_atom_idces,
            kv_idx=block_kv_atom_idces,
            kv_is_wrapped_mask=kv_mask,
        )
        raw_b_i |= dict(
            block_atom_pair_q_idces=block_q_atom_idces,
            block_atom_pair_kv_idces=block_kv_atom_idces,
            block_atom_pair_mask=block_atom_pair_mask,
        )

        # Compute features
        raw_batch["features"] = self.feature_factory.generate(raw_batch)

        return raw_batch
