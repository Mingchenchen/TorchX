"""Reads Chemical Components gz file and generates a CCD pickle file."""

import gzip
import pickle
import sys
from collections.abc import Sequence

import tqdm
from torchx.cpp import cif_dict


def main(argv: Sequence[str]) -> None:
  if len(argv) != 3:
    raise ValueError('Must specify input_file components.cif and output_file')

  _, input_file, output_file = argv

  print(f'Parsing {input_file}', flush=True)
  if input_file.endswith('.gz'):
    opener = gzip.open
  else:
    opener = open

  with opener(input_file, 'rb') as f:
    whole_file = f.read()
  result = {
      key: {k: tuple(v) for k, v in value.items()}
      for key, value in tqdm.tqdm(
          cif_dict.parse_multi_data_cif(whole_file).items(), disable=None
      )
  }
  assert len(result) == whole_file.count(b'data_')

  print(f'Writing {output_file}', flush=True)
  with open(output_file, 'wb') as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
  print('Done', flush=True)

if __name__ == '__main__':
  main(sys.argv)
