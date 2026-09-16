

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"

namespace {

namespace py = pybind11;

absl::flat_hash_map<int64_t, std::vector<int64_t>> IndicesGroupedByValue(
    absl::Span<const int64_t> values) {
  absl::flat_hash_map<int64_t, std::vector<int64_t>> group_indices;
  for (int64_t i = 0, e = values.size(); i < e; ++i) {
    group_indices[values[i]].push_back(i);
  }
  return group_indices;
}

constexpr char kIndicesGroupedByValue[] = R"(
Returns a map from value to a list of indices this value occupies.

E.g. indices_grouped_by_value([1, 1, 2, 3, 3, 1, 1]) returns:
{1: [0, 1, 5, 6], 2: [2], 3: [3, 4]}

Args:
  values: a list of values to group.
)";

}  // namespace

namespace torchx {

void RegisterModuleAggregation(py::module m) {
  m.def("indices_grouped_by_value", &IndicesGroupedByValue, py::arg("values"),
        py::doc(kIndicesGroupedByValue + 1),
        py::call_guard<py::gil_scoped_release>());
}

}  // namespace torchx
