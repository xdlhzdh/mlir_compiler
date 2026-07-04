#pragma once
// conversion_framework.h — P4 tier 3: pattern-based ONNX → StableHLO conversion framework
//
// Mirrors the key abstractions of MLIR's DialectConversion:
//   ConversionPattern  — per-op rewrite rule (match + rewrite)
//   ConversionTarget   — defines legal / illegal op sets
//   RewritePatternSet  — pattern collection with priority (benefit)
//   applyFullConversion — walks graph, applies patterns, verifies legality
//
// This is a teaching implementation; real MLIR DialectConversion handles
// regions, block arguments, type conversion, and rollback — omitted here
// for clarity.

#include "onnx_to_shlo_utils.h"
#include <memory>
#include <unordered_set>

namespace framework {

// ======================== LogicalResult ========================

struct LogicalResult {
  bool succeeded_;
  static LogicalResult success() { return {true}; }
  static LogicalResult failure() { return {false}; }
  bool succeeded() const { return succeeded_; }
  bool failed() const { return !succeeded_; }
};

// ======================== ConversionPattern ========================

class ConversionPattern {
public:
  virtual ~ConversionPattern() = default;

  // Which ONNX op_type this pattern handles
  virtual std::string target_op_type() const = 0;

  // Higher benefit → tried first when multiple patterns match
  virtual int benefit() const { return 0; }

  // Match-and-rewrite: return success() if the node was converted
  virtual LogicalResult
  matchAndRewrite(const onnx::NodeProto &node,
                  onnx2shlo::Context &ctx) const = 0;
};

// ======================== ConversionTarget ========================

class ConversionTarget {
  std::unordered_set<std::string> legal_;
  std::unordered_set<std::string> illegal_;

public:
  void addLegalOp(const std::string &op) { legal_.insert(op); }
  void addIllegalOp(const std::string &op) { illegal_.insert(op); }

  void addLegalDialect(const std::string &prefix) {
    legal_.insert("__dialect__" + prefix);
  }

  bool isLegal(const std::string &op) const {
    if (legal_.count(op)) return true;
    // Check dialect prefix
    auto dot = op.find('.');
    if (dot != std::string::npos)
      if (legal_.count("__dialect__" + op.substr(0, dot)))
        return true;
    return false;
  }

  bool isIllegal(const std::string &op) const {
    return illegal_.count(op) > 0;
  }
};

// ======================== RewritePatternSet ========================

class RewritePatternSet {
  std::vector<std::unique_ptr<ConversionPattern>> patterns_;

public:
  template <typename PatternT, typename... Args>
  void add(Args &&...args) {
    patterns_.push_back(
        std::make_unique<PatternT>(std::forward<Args>(args)...));
  }

  // Sort by benefit (descending)
  void sort() {
    std::sort(patterns_.begin(), patterns_.end(),
              [](const auto &a, const auto &b) {
                return a->benefit() > b->benefit();
              });
  }

  const ConversionPattern *findPattern(const std::string &op_type) const {
    for (auto &p : patterns_)
      if (p->target_op_type() == op_type)
        return p.get();
    return nullptr;
  }

  const std::vector<std::unique_ptr<ConversionPattern>> &patterns() const {
    return patterns_;
  }
};

// ======================== Full Conversion Engine ========================

struct ConversionStats {
  int total = 0;
  int converted = 0;
  int already_legal = 0;
  int failed = 0;
  std::vector<std::string> failed_ops;
};

// Walk graph nodes, apply matching pattern for each illegal op,
// then verify all produced ops are legal.
inline LogicalResult
applyFullConversion(const onnx::GraphProto &graph, ConversionTarget &target,
                    RewritePatternSet &patterns, onnx2shlo::Context &ctx,
                    ConversionStats &stats) {
  patterns.sort();

  for (auto &node : graph.node()) {
    ++stats.total;
    const auto &op = node.op_type();

    if (target.isLegal(op)) {
      ++stats.already_legal;
      continue;
    }

    auto *pattern = patterns.findPattern(op);
    if (!pattern) {
      ctx.warn("No pattern for illegal op: " + op);
      ++stats.failed;
      stats.failed_ops.push_back(op);
      ++ctx.skipped;
      continue;
    }

    auto result = pattern->matchAndRewrite(node, ctx);
    if (result.succeeded()) {
      ++stats.converted;
      ++ctx.converted;
    } else {
      ctx.error("Pattern failed for op: " + op);
      ++stats.failed;
      stats.failed_ops.push_back(op);
    }
  }

  return (stats.failed == 0) ? LogicalResult::success()
                              : LogicalResult::failure();
}

// Verify all ops in the produced function are legal
inline bool verifyLegality(const shlo::FuncOp &func,
                           const ConversionTarget &target) {
  bool ok = true;
  for (auto &op : func.body) {
    if (!target.isLegal(op.mnemonic)) {
      std::cerr << "  [ILLEGAL] " << op.mnemonic << "\n";
      ok = false;
    }
  }
  return ok;
}

} // namespace framework
