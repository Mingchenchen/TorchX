#include "torchfold/processing/mkdssp_pybind.h"

#include <filesystem>

#include <cif++/file.hpp>
#include <cif++/pdb.hpp>
#include <dssp.hpp>
#include <sstream>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace torchfold {
namespace py = pybind11;

void RegisterModuleMkdssp(pybind11::module m) {
  py::module site = py::module::import("site");
  py::list paths = py::cast<py::list>(site.attr("getsitepackages")());
  // Find the first path that contains the libcifpp components.cif file.
  bool found = false;
  for (const auto& py_path : paths) {
    auto path_str =
        std::filesystem::path(py::cast<absl::string_view>(py_path)) /
        "share/libcifpp/components.cif";
    if (std::filesystem::exists(path_str)) {
      setenv("LIBCIFPP_DATA_DIR", path_str.parent_path().c_str(), 0);
      found = true;
      break;
    }
  }
  if (!found) {
    throw py::type_error("Could not find the libcifpp components.cif file.");
  }
  m.def(
      "get_dssp",
      [](absl::string_view mmcif, int model_no,
         int min_poly_proline_stretch_length,
         bool calculate_surface_accessibility) {
        cif::file cif_file(mmcif.data(), mmcif.size());
        dssp result(cif_file.front(), model_no, min_poly_proline_stretch_length,
                    calculate_surface_accessibility);
        std::stringstream sstream;
        result.write_legacy_output(sstream);
        return sstream.str();
      },
      py::arg("mmcif"), py::arg("model_no") = 1,
      py::arg("min_poly_proline_stretch_length") = 3,
      py::arg("calculate_surface_accessibility") = false,
      py::doc("Gets secondary structure from an mmCIF file."));
}

}  // namespace torchfold
