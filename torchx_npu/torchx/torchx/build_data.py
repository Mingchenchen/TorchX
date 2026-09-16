

"""Script for building intermediate data."""

import pathlib
import site
from importlib import resources

import torchx.constants.converters
from torchx.constants.converters import ccd_pickle_gen
from torchx.constants.converters import chemical_component_sets_gen


def build_data():
  """Builds intermediate data."""
  for site_path in site.getsitepackages():
    path = pathlib.Path(site_path) / 'share/libcifpp/components.cif'
    if path.exists():
      cif_path = path
      break
  else:
    raise ValueError('Could not find components.cif')

  out_root = resources.files(torchx.constants.converters)
  ccd_pickle_path = out_root.joinpath('ccd.pickle')
  chemical_component_sets_pickle_path = out_root.joinpath(
      'chemical_component_sets.pickle'
  )
  ccd_pickle_gen.main(['', str(cif_path), str(ccd_pickle_path)])
  chemical_component_sets_gen.main(
      ['', str(chemical_component_sets_pickle_path)]
  )
