#include "torchfold/processing/mkdssp_pybind.h"
#include "torchfold/parsers/cpp/cif_dict_pybind.h"
#include "torchfold/structure/cpp/aggregation_pybind.h"
#include "torchfold/structure/cpp/membership_pybind.h"
#include "torchfold/structure/cpp/mmcif_atom_site_pybind.h"
#include "torchfold/structure/cpp/mmcif_layout_pybind.h"
#include "torchfold/structure/cpp/mmcif_struct_conn_pybind.h"
#include "torchfold/structure/cpp/mmcif_utils_pybind.h"
#include "torchfold/structure/cpp/string_array_pybind.h"
#include "pybind11/pybind11.h"

namespace torchfold {
namespace {

// Include all modules as submodules to simplify building.
PYBIND11_MODULE(cpp, m) {
  RegisterModuleCifDict(m.def_submodule("cif_dict"));
  RegisterModuleMmcifLayout(m.def_submodule("mmcif_layout"));
  RegisterModuleMmcifStructConn(m.def_submodule("mmcif_struct_conn"));
  RegisterModuleMembership(m.def_submodule("membership"));
  RegisterModuleMmcifUtils(m.def_submodule("mmcif_utils"));
  RegisterModuleAggregation(m.def_submodule("aggregation"));
  RegisterModuleStringArray(m.def_submodule("string_array"));
  RegisterModuleMmcifAtomSite(m.def_submodule("mmcif_atom_site"));
  RegisterModuleMkdssp(m.def_submodule("mkdssp"));
}

}  // namespace
}  // namespace torchfold
