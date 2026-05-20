#pragma once
// memory_ir.h — P12 (15_memory_planning): simplified IR for memory planning.
//
// Models memory management concepts critical to AI compiler efficiency:
//
//   BufferReq     — memory requirement for a single tensor/buffer
//   LiveInterval  — [first_use, last_use] liveness range for each buffer
//   MemPool       — arena-style memory pool with offset-based allocation
//   AllocPlan     — final memory plan: buffer → (pool, offset, size)
//
// Header-only, pure C++17, no external dependencies.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace mem_ir {

// ======================== Buffer Request ========================

struct BufferReq {
  int id = -1;
  std::string name;
  int64_t size_bytes = 0;
  int alignment = 64;
  bool is_input = false;
  bool is_output = false;
  bool is_weight = false;

  int64_t aligned_size() const {
    return (size_bytes + alignment - 1) / alignment * alignment;
  }
};

// ======================== Live Interval ========================

struct LiveInterval {
  int buf_id = -1;
  int first_use = -1;
  int last_use = -1;

  bool overlaps(const LiveInterval &other) const {
    return first_use <= other.last_use && other.first_use <= last_use;
  }

  int length() const { return last_use - first_use + 1; }
};

// ======================== Memory Pool ========================

struct Allocation {
  int buf_id;
  int64_t offset;
  int64_t size;
};

struct MemPool {
  std::string name;
  int64_t total_size = 0;
  std::vector<Allocation> allocs;

  void print(std::ostream &os) const {
    os << "  Pool \"" << name << "\": " << total_size << " bytes ("
       << total_size / 1024 << " KB)\n";
    for (auto &a : allocs) {
      os << "    buf[" << a.buf_id << "] @ offset " << a.offset
         << " size " << a.size << " [" << a.offset
         << ", " << a.offset + a.size << ")\n";
    }
  }
};

// ======================== Op (for scheduling) ========================

struct Op {
  int id = -1;
  std::string name;
  std::vector<int> input_bufs;
  std::vector<int> output_bufs;
  int64_t compute_cost = 0;
  bool is_elementwise = false;
};

// ======================== Execution Graph ========================

struct ExecGraph {
  std::vector<Op> ops;
  std::vector<BufferReq> buffers;

  Op &add_op(const std::string &name) {
    ops.push_back({static_cast<int>(ops.size()), name, {}, {}, 0, false});
    return ops.back();
  }

  int add_buffer(const std::string &name, int64_t size,
                 bool is_in = false, bool is_out = false,
                 bool is_weight = false) {
    int id = static_cast<int>(buffers.size());
    buffers.push_back({id, name, size, 64, is_in, is_out, is_weight});
    return id;
  }
};

}  // namespace mem_ir
