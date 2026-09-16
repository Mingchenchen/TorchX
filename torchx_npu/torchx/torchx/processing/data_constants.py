

"""Constants shared across modules in the TorchFold data pipeline."""

from torchx.constants import residue_names

MSA_GAP_IDX = residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP.index(
    '-'
)

# Feature groups.
NUM_SEQ_NUM_RES_MSA_FEATURES = ('msa', 'msa_mask', 'deletion_matrix')
NUM_SEQ_MSA_FEATURES = ('msa_species_identifiers',)
TEMPLATE_FEATURES = (
    'template_aatype',
    'template_atom_positions',
    'template_atom_mask',
)
MSA_PAD_VALUES = {'msa': MSA_GAP_IDX, 'msa_mask': 1, 'deletion_matrix': 0}
