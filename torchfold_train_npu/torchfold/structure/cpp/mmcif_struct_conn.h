#ifndef TORCHFOLD_SRC_TORCHFOLD_STRUCTURE_PYTHON_MMCIF_STRUCT_CONN_H_
#define TORCHFOLD_SRC_TORCHFOLD_STRUCTURE_PYTHON_MMCIF_STRUCT_CONN_H_

#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "torchfold/parsers/cpp/cif_dict_lib.h"

namespace torchfold {

// Returns a pair of atom indices for each row in the bonds table (aka
// _struct_conn). The indices are simple 0-based indexes into the columns of
// the _atom_site table in the input mmCIF, and do not necessarily correspond
// to the values in _atom_site.id, or any other column.
absl::StatusOr<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>
GetBondAtomIndices(const CifDict& mmcif, absl::string_view model_id);

}  // namespace torchfold

#endif  // TORCHFOLD_SRC_TORCHFOLD_STRUCTURE_PYTHON_MMCIF_STRUCT_CONN_H_
