#ifndef TORCHFOLD_SRC_TORCHFOLD_STRUCTURE_PYTHON_MMCIF_ALTLOCS_H_
#define TORCHFOLD_SRC_TORCHFOLD_STRUCTURE_PYTHON_MMCIF_ALTLOCS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "torchfold/structure/cpp/mmcif_layout.h"

namespace torchfold {

// Returns the list of indices that should be kept after resolving alt-locs.
// 1) Partial Residue. Each cycle of alt-locs are resolved separately with the
//    highest occupancy alt-loc. Tie-breaks are resolved alphabetically. See
//    tests for examples.
// 2) Whole Residue. These are resolved in two passes.
//    a) The residue with the highest occupancy is chosen.
//    b) The locations for a given residue are resolved.
//    All tie-breaks are resolved alphabetically. See tests for examples.
//
// Preconditions: layout and comp_ids, alt_ids, occupancies are all from same
// mmCIF file and chain_indices are monotonically increasing and less than
// layout.num_chains().
//
// comp_ids from '_atom_site.label_comp_id'.
// alt_ids from '_atom_site.label_alt_id'.
// occupancies from '_atom_site.occupancy'.
std::vector<std::uint64_t> ResolveMmcifAltLocs(
    const MmcifLayout& layout, absl::Span<const std::string> comp_ids,
    absl::Span<const std::string> atom_ids,
    absl::Span<const std::string> alt_ids,
    absl::Span<const std::string> occupancies,
    absl::Span<const std::size_t> chain_indices);

}  // namespace torchfold

#endif  // TORCHFOLD_SRC_TORCHFOLD_STRUCTURE_PYTHON_MMCIF_ALTLOCS_H_
