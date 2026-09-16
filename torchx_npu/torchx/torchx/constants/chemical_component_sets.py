"""Sets of chemical components."""

import pickle
from typing import Final

from torchx.common import resources

_CCD_SETS_CCD_PICKLE_FILE = resources.filename(
    resources.ROOT / 'constants/converters/chemical_component_sets.pickle'
)

_CCD_SET = pickle.load(open(_CCD_SETS_CCD_PICKLE_FILE, 'rb'))

# Glycan (or 'Saccharide') ligands.
# _chem_comp.type containing 'saccharide' and 'linking' (when lower-case).
GLYCAN_LINKING_LIGANDS: Final[frozenset[str]] = _CCD_SET['glycans_linking']

# _chem_comp.type containing 'saccharide' and not 'linking' (when lower-case).
GLYCAN_OTHER_LIGANDS: Final[frozenset[str]] = _CCD_SET['glycans_other']

# Each of these molecules appears in over 1k PDB structures, are used to
# facilitate crystallization conditions, but do not have biological relevance.
COMMON_CRYSTALLIZATION_AIDS: Final[frozenset[str]] = frozenset({
    'SO4', 'GOL', 'EDO', 'PO4', 'ACT', 'PEG', 'DMS', 'TRS', 'PGE', 'PG4', 'FMT',
    'EPE', 'MPD', 'MES', 'CD', 'IOD',
})  # pyformat: disable
