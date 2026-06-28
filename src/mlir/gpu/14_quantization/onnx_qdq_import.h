#pragma once
// onnx_qdq_import.h — load ONNX QDQ graphs into quant_ir (P11).

#include "quant_ir.h"

#include <iostream>
#include <string>

namespace quant_ir {

struct QdqImportStats {
  int quantize_linear = 0;
  int dequantize_linear = 0;
  int matmul = 0;
  int other = 0;
};

Graph load_onnx_qdq(const std::string &path, QdqImportStats &stats,
                    std::ostream &log = std::cout);

}  // namespace quant_ir
