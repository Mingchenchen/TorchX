from collections.abc import Mapping

import numpy as np
from torchfold.cpp import string_array

from torchfold.constants import mmcif_names
from torchfold.structure.tables import Chains, Residues


class _ChainResBuilder:
    """Class for incrementally building chain and residue tables."""

    def __init__(
            self, *, chain_key_by_chain_id: Mapping[str, int],
            entity_id_by_chain_id: Mapping[str, str],
            chain_type_by_entity_id: Mapping[str, str],
            entity_desc_by_entity_id: Mapping[str, str],
            fix_mse_residues: bool,
            fix_unknown_dna: bool,
    ):
        self.chain_key, self.chain_id, self.chain_type = [], [], []
        self.chain_auth_asym_id, self.chain_entity_id, self.chain_entity_desc = [], [], []
        self.res_key, self.res_chain_key, self.res_id = [], [], []
        self.res_name, self.res_auth_seq_id, self.res_insertion_code = [], [], []

        self.chain_key_by_chain_id = chain_key_by_chain_id
        self.entity_id_by_chain_id = entity_id_by_chain_id
        self.chain_type_by_entity_id = chain_type_by_entity_id
        self.entity_desc_by_entity_id = entity_desc_by_entity_id
        self.key_for_res: dict[tuple[str, str, str, str], int] = {}
        self._fix_mse_residues = fix_mse_residues
        self._fix_unknown_dna = fix_unknown_dna

    def add_residues(self, *, chain_ids, chain_auth_asym_ids, res_ids, res_names, res_auth_seq_ids, res_ins_codes):
        """Adds a residue (and its chain) to the tables."""
        if chain_ids.size == 0:
            return
        chain_ids_with_prev = np.concatenate([self.chain_id[-1] if self.chain_id else None], chain_ids)
        chain_change_mask = chain_ids_with_prev[:-1] != chain_ids_with_prev[1:]
        chain_change_ids = chain_ids[chain_change_mask]
        chain_keys = string_array.remap(chain_change_ids, self.chain_key_by_chain_id, inplace=False)
        self.chain_key.extend(chain_keys)
        self.chain_id.extend(chain_change_ids)
        self.chain_auth_asym_id.extend(chain_auth_asym_ids[chain_change_mask])
        chain_entity_id = string_array.remap(chain_change_ids, self.entity_id_by_chain_id, inplace=False)
        self.chain_entity_id.extend(chain_entity_id)
        chain_type = string_array.remap(chain_entity_id, self.chain_type_by_entity_id, inplace=False)
        self.chain_type.extend(chain_type)
        chain_entity_desc = string_array.remap(chain_entity_id, self.entity_desc_by_entity_id, inplace=False)
        self.chain_entity_desc.extend(chain_entity_desc)
        num_prev_res = len(self.res_id)
        res_keys = np.arange(num_prev_res, num_prev_res + len(res_ids))
        res_iter = zip(chain_ids, res_auth_seq_ids, res_names, res_ins_codes, strict=True)
        key_for_res_update = {res_unique_id: res_key for res_key, res_unique_id in enumerate(res_iter, num_prev_res)}
        self.key_for_res.update(key_for_res_update)
        self.res_key.extend(res_keys)
        self.res_chain_key.extend(string_array.remap(chain_ids, self.chain_key_by_chain_id, inplace=False))
        self.res_id.extend(res_ids)
        self.res_name.extend(res_names)
        self.res_auth_seq_id.extend(res_auth_seq_ids)
        self.res_insertion_code.extend(res_ins_codes)

    def make_chains_table(self) -> Chains:
        chain_key = np.array(self.chain_key, dtype=np.int64)
        if not np.all(chain_key[:-1] <= chain_key[1:]):
            order = np.argsort(self.chain_key, kind='stable')
            return Chains(
                key=chain_key[order], id=np.array(self.chain_id, dtype=object)[order],
                type=np.array(self.chain_type, dtype=object)[order],
                auth_asym_id=np.array(self.chain_auth_asym_id, dtype=object)[order],
                entity_id=np.array(self.chain_entity_id, dtype=object)[order],
                entity_desc=np.array(self.chain_entity_desc, dtype=object)[order],
            )
        return Chains(
            key=chain_key, id=np.array(self.chain_id, dtype=object),
            type=np.array(self.chain_type, dtype=object),
            auth_asym_id=np.array(self.chain_auth_asym_id, dtype=object),
            entity_id=np.array(self.chain_entity_id, dtype=object),
            entity_desc=np.array(self.chain_entity_desc, dtype=object),
        )

    def make_residues_table(self) -> Residues:
        res_name = np.array(self.res_name, dtype=object)
        res_chain_key = np.array(self.res_chain_key, dtype=np.int64)
        if self._fix_mse_residues:
            string_array.remap(res_name, mapping={'MSE': 'MET'}, inplace=True)
        if self._fix_unknown_dna:
            dna_chain_mask = (np.array(self.chain_type, dtype=object) == mmcif_names.DNA_CHAIN)
            dna_chain_key = np.array(self.chain_key, dtype=object)[dna_chain_mask]
            res_name[(res_name == 'N') & np.isin(res_chain_key, dna_chain_key)] = 'DN'
        if not np.all(res_chain_key[:-1] <= res_chain_key[1:]):
            order = np.argsort(res_chain_key, kind='stable')
            return Residues(
                key=np.array(self.res_key, dtype=np.int64)[order], chain_key=res_chain_key[order],
                id=np.array(self.res_id, dtype=np.int32)[order], name=res_name[order],
                auth_seq_id=np.array(self.res_auth_seq_id, dtype=object)[order],
                insertion_code=np.array(self.res_insertion_code, dtype=object)[order],
            )
        return Residues(
            key=np.array(self.res_key, dtype=np.int64), chain_key=res_chain_key,
            id=np.array(self.res_id, dtype=np.int32), name=res_name,
            auth_seq_id=np.array(self.res_auth_seq_id, dtype=object),
            insertion_code=np.array(self.res_insertion_code, dtype=object),
        )
