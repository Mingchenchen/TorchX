import collections
import datetime
from collections.abc import Mapping, Sequence

from torchfold.structure import mmcif
from torchfold.structure.parsing import table_builder
from torchfold.structure.internal.constants import COORDS_DECIMAL_PLACES


def structure_to_mmcif_header(struc) -> Mapping[str, Sequence[str]]:
    raw_mmcif = collections.defaultdict(list)
    raw_mmcif['data_'] = [struc.name.replace(' ', '-')]
    raw_mmcif['_entry.id'] = [struc.name]

    if struc.release_date is not None:
        date = [datetime.datetime.strftime(struc.release_date, '%Y-%m-%d')]
        raw_mmcif['_pdbx_audit_revision_history.revision_date'] = date
        raw_mmcif['_pdbx_database_status.recvd_initial_deposition_date'] = date

    if struc.resolution is not None:
        raw_mmcif['_refine.ls_d_res_high'] = ['%.2f' % struc.resolution]

    if struc.structure_method is not None:
        for method in struc.structure_method.split(','):
            raw_mmcif['_exptl.method'].append(method)

    if struc.bioassembly_data is not None:
        raw_mmcif.update(struc.bioassembly_data.to_mmcif_dict())

    # Populate chemical components data for all residues of this Structure.
    if struc.chemical_components_data:
        raw_mmcif.update(struc.chemical_components_data.to_mmcif_dict())

    # Add _software table to store version number used to generate mmCIF.
    # Only required data items are used (+ _software.version).
    raw_mmcif['_software.pdbx_ordinal'] = ['1']
    raw_mmcif['_software.name'] = ['TorchFold Structure Class']
    raw_mmcif['_software.version'] = [struc.version]
    raw_mmcif['_software.classification'] = ['other']  # Required.

    return raw_mmcif


def structure_to_mmcif_dict(
        struc,
        *,
        coords_decimal_places: int = COORDS_DECIMAL_PLACES,
) -> mmcif.Mmcif:
    """Returns an Mmcif representing the structure."""
    header = structure_to_mmcif_header(struc)
    sequence_tables = table_builder.to_mmcif_sequence_and_entity_tables(
        struc.chains_table, struc.residues_table, struc.atoms_table.res_key
    )
    atom_and_bond_tables = table_builder.to_mmcif_atom_site_and_bonds_table(
        chains=struc.chains_table,
        residues=struc.residues_table,
        atoms=struc.atoms_table,
        bonds=struc.bonds_table,
        coords_decimal_places=coords_decimal_places,
    )
    return mmcif.Mmcif({**header, **sequence_tables, **atom_and_bond_tables})


def structure_to_mmcif_string(
        struc,
        *,
        coords_decimal_places: int = COORDS_DECIMAL_PLACES
) -> str:
    """Returns an mmCIF string representing the structure.

           Args:
             coords_decimal_places: The number of decimal places to keep for atom
               coordinates, including trailing zeros.
           """
    return structure_to_mmcif_dict(
        struc, coords_decimal_places=coords_decimal_places
    ).to_string()
