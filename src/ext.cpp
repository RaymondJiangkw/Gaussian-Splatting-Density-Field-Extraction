#include <torch/extension.h>
#include "estimate_density.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("estimate_density", &EstimateDensityCUDA);
}