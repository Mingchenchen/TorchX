

#include "torchx/data/cpp/msa_profile_pybind.h"
#include "torchx/processing/mkdssp_pybind.h"
#include "torchx/parsers/cpp/cif_dict_pybind.h"
#include "torchx/parsers/cpp/fasta_iterator_pybind.h"
#include "torchx/parsers/cpp/msa_conversion_pybind.h"
#include "torchx/structure/cpp/aggregation_pybind.h"
#include "torchx/structure/cpp/membership_pybind.h"
#include "torchx/structure/cpp/mmcif_atom_site_pybind.h"
#include "torchx/structure/cpp/mmcif_layout_pybind.h"
#include "torchx/structure/cpp/mmcif_struct_conn_pybind.h"
#include "torchx/structure/cpp/mmcif_utils_pybind.h"
#include "torchx/structure/cpp/string_array_pybind.h"
#include "pybind11/pybind11.h"

namespace torchx {
namespace {

// Include all modules as submodules to simplify building.
PYBIND11_MODULE(cpp, m) {
  RegisterModuleCifDict(m.def_submodule("cif_dict"));
  RegisterModuleFastaIterator(m.def_submodule("fasta_iterator"));
  RegisterModuleMsaConversion(m.def_submodule("msa_conversion"));
  RegisterModuleMmcifLayout(m.def_submodule("mmcif_layout"));
  RegisterModuleMmcifStructConn(m.def_submodule("mmcif_struct_conn"));
  RegisterModuleMembership(m.def_submodule("membership"));
  RegisterModuleMmcifUtils(m.def_submodule("mmcif_utils"));
  RegisterModuleAggregation(m.def_submodule("aggregation"));
  RegisterModuleStringArray(m.def_submodule("string_array"));
  RegisterModuleMmcifAtomSite(m.def_submodule("mmcif_atom_site"));
  RegisterModuleMkdssp(m.def_submodule("mkdssp"));
  RegisterModuleMsaProfile(m.def_submodule("msa_profile"));
}

}  // namespace
}  // namespace torchx
