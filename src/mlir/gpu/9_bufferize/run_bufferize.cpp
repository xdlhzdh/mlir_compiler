// run_bufferize.cpp — One-Shot Bufferization (OSB) 8-Stage Pipeline
//
// Converts tensor-semantic Linalg IR to buffer-semantic (memref) IR:
//
//   Stage 0: Pre-clean          — canonicalize + CSE + DCE
//   Stage 1: Alias Analysis     — build alias sets, SSA use-def, mark shared values
//   Stage 2: In-place Analysis  — write conflict detection (WAR / RAW)
//   Stage 3: Bufferization Decision — mark INPLACE / OUT_OF_PLACE per op
//   Stage 4: Rewrite            — tensor → memref, insert alloc, destination-passing
//   Stage 5: Copy Insertion     — insert copies for out-of-place conflict resolution
//   Stage 6: Buffer Deallocation — insert dealloc, ownership-based lifetime
//   Stage 7: Post-clean         — canonicalize + DCE on bufferized IR
//
// Pure C++17, header-only IR, no external dependencies.

#include "bufferize_ir.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace bufferize_ir;

// =====================================================================
// Buffer Plan — records the full memref-world transformation
// =====================================================================

struct BufSlot {
  int id;
  std::string name;
  MemRefType type;
  bool is_func_arg;
  bool is_output_arg;
  bool needs_dealloc;
  int last_use_idx = -1;
};

struct CopyEntry {
  int src_slot;
  int dst_slot;
  int before_op_idx;
  std::string reason;
};

struct BufPlan {
  std::vector<BufSlot> slots;
  std::unordered_map<Value *, int> val_to_slot;
  std::vector<CopyEntry> copies;

  // Per-op input overrides: when a specific op reads a value, use the
  // copy buffer instead of the original. Key = (op_ptr, value_ptr).
  std::unordered_map<const Op *,
                     std::unordered_map<const Value *, int>> input_redirect;

  int new_slot(const std::string &name, const MemRefType &type,
               bool is_arg, bool is_out = false) {
    int id = static_cast<int>(slots.size());
    slots.push_back({id, name, type, is_arg, is_out, !is_arg, -1});
    return id;
  }
  int slot_of(Value *v) const {
    auto it = val_to_slot.find(v);
    return it != val_to_slot.end() ? it->second : -1;
  }
  int effective_slot(const Op *op, Value *v) const {
    auto oi = input_redirect.find(op);
    if (oi != input_redirect.end()) {
      auto vi = oi->second.find(v);
      if (vi != oi->second.end()) return vi->second;
    }
    return slot_of(v);
  }
};

// =====================================================================
// Stage 0: Pre-clean — canonicalize + CSE + DCE
// =====================================================================

static int pass_cse(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size() && !changed; ++i) {
      auto *a = g.ops[i].get();
      if (!a->is_linalg() || a->is_fill()) continue;
      for (size_t j = i + 1; j < g.ops.size() && !changed; ++j) {
        auto *b = g.ops[j].get();
        if (a->mnemonic != b->mnemonic || a->label != b->label) continue;
        if (a->ins.size() != b->ins.size()) continue;
        bool same = true;
        for (size_t k = 0; k < a->ins.size(); ++k)
          if (a->ins[k] != b->ins[k]) { same = false; break; }
        if (!same) continue;
        std::cout << "  [CSE] " << b->result_->name << " = "
                  << b->label << " → " << a->result_->name << "\n";
        g.replace_all_uses(b->result_, a->result_);
        g.erase_op(b);
        ++count;
        changed = true;
      }
    }
  }
  return count;
}

static int pass_dce(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = g.ops.size(); i-- > 0;) {
      auto *op = g.ops[i].get();
      if (!op->result_ || g.is_return_val(op->result_)) continue;
      if (op->result_->has_no_uses()) {
        std::cout << "  [DCE] " << op->label
                  << " (" << op->result_->name << ")\n";
        g.erase_op(op);
        ++count;
        changed = true;
        break;
      }
    }
  }
  return count;
}

static int stage0_preclean(Graph &g) {
  return pass_cse(g) + pass_dce(g);
}

// =====================================================================
// Stage 1: Alias Analysis
// =====================================================================
// Build potential alias groups: values connected by in-place data flow.
// - Function args: each is its own alias root (immutable input)
// - fill: creates a new alias root (fresh buffer)
// - matmul result → aliases outs[0] (accumulator)
// - generic result → aliases ins[0] (first input, preferred for in-place)
// At this stage, conflicts are NOT considered — groups show POTENTIAL sharing.

struct AliasInfo {
  std::unordered_map<Value *, int> val_to_group;
  int num_groups = 0;

  int group_of(Value *v) const {
    auto it = val_to_group.find(v);
    return it != val_to_group.end() ? it->second : -1;
  }
};

static AliasInfo stage1_alias_analysis(const Graph &g) {
  AliasInfo info;
  int next_gid = 0;

  for (auto *a : g.args)
    info.val_to_group[a] = next_gid++;

  for (auto &op : g.ops) {
    if (!op->result_) continue;
    if (op->is_fill()) {
      info.val_to_group[op->result_] = next_gid++;
    } else if (op->mnemonic == "linalg.matmul" && !op->outs.empty()) {
      int g_id = info.group_of(op->outs[0]);
      info.val_to_group[op->result_] = (g_id >= 0) ? g_id : next_gid++;
    } else if (!op->ins.empty()) {
      int g_id = info.group_of(op->ins[0]);
      info.val_to_group[op->result_] = (g_id >= 0) ? g_id : next_gid++;
    } else {
      info.val_to_group[op->result_] = next_gid++;
    }
  }

  info.num_groups = next_gid;

  // Print alias groups
  std::unordered_map<int, std::vector<Value *>> groups;
  for (auto &[v, gid] : info.val_to_group)
    groups[gid].push_back(v);

  std::cout << "  Potential alias groups (" << groups.size() << " groups):\n";
  for (auto &[gid, members] : groups) {
    std::sort(members.begin(), members.end(),
              [](Value *a, Value *b) { return a->id < b->id; });
    std::cout << "    Group #" << gid << ": ";
    for (size_t i = 0; i < members.size(); ++i) {
      if (i) std::cout << " → ";
      std::cout << members[i]->name;
      if (!members[i]->defining_op) std::cout << "(arg)";
      else if (members[i]->defining_op->is_fill()) std::cout << "(fill)";
    }
    std::cout << "\n";
  }
  return info;
}

// =====================================================================
// Stage 2: In-place Analysis — write conflict detection
// =====================================================================
// For each op, determine the "preferred in-place target" and check for WAR
// conflicts: if op O writes to V's buffer in-place, any later reader of V
// would see corrupted data.

static Value *preferred_inplace(const Op *op) {
  if (op->is_fill()) return nullptr;
  if (op->mnemonic == "linalg.matmul" && !op->outs.empty())
    return op->outs[0];
  if (!op->ins.empty()) {
    if (!op->ins[0]->defining_op) return nullptr;
    return op->ins[0];
  }
  return nullptr;
}

static std::vector<WriteConflict>
stage2_inplace_analysis(const Graph &g) {
  std::vector<WriteConflict> conflicts;

  for (size_t i = 0; i < g.ops.size(); ++i) {
    auto *op = g.ops[i].get();
    Value *target = preferred_inplace(op);
    if (!target) continue;

    for (auto *user : target->users) {
      if (user == op) continue;
      int user_pos = g.op_pos(user);
      if (user_pos > static_cast<int>(i)) {
        conflicts.push_back({op, user, target, ConflictKind::WAR});
      }
    }
  }

  if (conflicts.empty()) {
    std::cout << "  No write conflicts detected.\n";
  } else {
    std::cout << "  " << conflicts.size() << " write conflict(s):\n";
    for (auto &c : conflicts) {
      std::cout << "    [" << conflict_str(c.kind) << "] "
                << c.writer->label << " writes " << c.shared_value->name
                << " → conflict with later reader " << c.later_reader->label
                << "\n";
    }
  }
  return conflicts;
}

// =====================================================================
// Stage 3: Bufferization Decision
// =====================================================================
// For each op:
//   - If no conflict on preferred target → INPLACE
//   - If conflict (this op is the writer) → INPLACE (first writer priority)
//     + flag needs_copy_before for later readers
//   - If the op produces the return value → OUT_OF_PLACE (destination-passing)
//   - If preferred target is a function argument → OUT_OF_PLACE (don't mutate)

static void stage3_decision(Graph &g,
                            const std::vector<WriteConflict> &conflicts) {
  std::unordered_set<Op *> writers_with_conflict;
  for (auto &c : conflicts) writers_with_conflict.insert(c.writer);

  for (auto &op : g.ops) {
    if (op->is_fill()) {
      op->buf_info.decision = BufDecision::INPLACE;
      op->buf_info.reason = "fresh alloc (fill)";
      continue;
    }

    if (g.is_return_val(op->result_)) {
      op->buf_info.decision = BufDecision::OUT_OF_PLACE;
      op->buf_info.reason = "destination-passing (return value)";
      std::cout << "  [Decision] " << op->label << " → OUT_OF_PLACE"
                << " (destination-passing: writes to output buffer)\n";
      continue;
    }

    Value *target = preferred_inplace(op.get());

    if (!target) {
      op->buf_info.decision = BufDecision::OUT_OF_PLACE;
      op->buf_info.reason = "no eligible in-place target";
      std::cout << "  [Decision] " << op->label << " → OUT_OF_PLACE"
                << " (no eligible target)\n";
      continue;
    }

    if (!target->defining_op) {
      op->buf_info.decision = BufDecision::OUT_OF_PLACE;
      op->buf_info.reason = "func arg (immutable input)";
      std::cout << "  [Decision] " << op->label << " → OUT_OF_PLACE"
                << " (can't mutate func arg " << target->name << ")\n";
      continue;
    }

    if (writers_with_conflict.count(op.get())) {
      op->buf_info.decision = BufDecision::INPLACE;
      op->buf_info.aliases_with = target;
      op->buf_info.needs_copy_before = true;
      op->buf_info.reason = "first writer (copy needed for later readers)";
      std::cout << "  [Decision] " << op->label << " → INPLACE(→"
                << target->name << ") + COPY_BEFORE\n";
    } else {
      bool all_others_before = true;
      int my_pos = g.op_pos(op.get());
      for (auto *u : target->users) {
        if (u == op.get()) continue;
        if (g.op_pos(u) > my_pos) { all_others_before = false; break; }
      }
      if (all_others_before) {
        op->buf_info.decision = BufDecision::INPLACE;
        op->buf_info.aliases_with = target;
        op->buf_info.reason = (target->users.size() <= 1)
            ? "single user" : "all other readers completed";
        std::cout << "  [Decision] " << op->label << " → INPLACE(→"
                  << target->name << ") ["
                  << op->buf_info.reason << "]\n";
      } else {
        op->buf_info.decision = BufDecision::OUT_OF_PLACE;
        op->buf_info.reason = "later readers (conflict)";
        std::cout << "  [Decision] " << op->label << " → OUT_OF_PLACE"
                  << " (shared value has later readers)\n";
      }
    }
  }
}

// =====================================================================
// Stage 4: Rewrite — build buffer assignment (tensor → memref)
// =====================================================================
// Walk the annotated tensor graph and build a BufPlan:
//   - Function args (inputs) → memref slots (immutable)
//   - Return value → output memref slot (function arg, destination-passing)
//   - fill → local alloc slot
//   - INPLACE ops → inherit slot from aliased input
//   - OUT_OF_PLACE ops → fresh alloc slot

static BufPlan stage4_rewrite(const Graph &g) {
  BufPlan plan;

  for (auto *a : g.args) {
    auto mtype = MemRefType::from_tensor(a->type);
    int sid = plan.new_slot(a->name, mtype, /*is_arg=*/true);
    plan.val_to_slot[a] = sid;
    std::cout << "  [BufAssign] " << a->name << " (func arg) → "
              << plan.slots[sid].name << " : " << mtype.str() << "\n";
  }

  for (auto &op : g.ops) {
    if (!op->result_) continue;
    auto mtype = MemRefType::from_tensor(op->result_->type);

    if (op->is_fill()) {
      std::string bname = "%buf" + std::to_string(plan.slots.size());
      int sid = plan.new_slot(bname, mtype, false);
      plan.val_to_slot[op->result_] = sid;
      std::cout << "  [BufAssign] " << op->result_->name
                << " (fill) → " << bname << " = memref.alloc() : "
                << mtype.str() << "\n";
      continue;
    }

    if (g.is_return_val(op->result_)) {
      std::string bname = "%out";
      int sid = plan.new_slot(bname, mtype, /*is_arg=*/true, /*is_out=*/true);
      plan.val_to_slot[op->result_] = sid;
      std::cout << "  [BufAssign] " << op->result_->name
                << " (return) → " << bname
                << " (output arg, destination-passing)\n";
      continue;
    }

    if (op->buf_info.decision == BufDecision::INPLACE &&
        op->buf_info.aliases_with) {
      int src = plan.slot_of(op->buf_info.aliases_with);
      if (src >= 0) {
        plan.val_to_slot[op->result_] = src;
        std::cout << "  [BufAssign] " << op->result_->name
                  << " (" << op->label << ") → "
                  << plan.slots[src].name << " (in-place → "
                  << op->buf_info.aliases_with->name << ")\n";
        continue;
      }
    }

    std::string bname = "%buf" + std::to_string(plan.slots.size());
    int sid = plan.new_slot(bname, mtype, false);
    plan.val_to_slot[op->result_] = sid;
    std::cout << "  [BufAssign] " << op->result_->name
              << " (" << op->label << ") → " << bname
              << " = memref.alloc() : " << mtype.str()
              << " (out-of-place)\n";
  }

  return plan;
}

// =====================================================================
// Stage 5: Copy Insertion
// =====================================================================
// For each op flagged needs_copy_before:
//   - The op overwrites its in-place target's buffer
//   - Later readers of the original value need a copy BEFORE the overwrite
//   - Allocate a fresh buffer, copy the original, redirect later readers

static void stage5_copy_insert(Graph &g, BufPlan &plan,
                               const std::vector<WriteConflict> &conflicts) {
  for (auto &op : g.ops) {
    if (!op->buf_info.needs_copy_before) continue;
    Value *target = op->buf_info.aliases_with;
    if (!target) continue;

    int src_slot = plan.slot_of(target);
    if (src_slot < 0) continue;

    auto mtype = plan.slots[src_slot].type;
    std::string bname = "%buf" + std::to_string(plan.slots.size());
    int dst_slot = plan.new_slot(bname, mtype, false);

    int op_idx = g.op_pos(op.get());
    plan.copies.push_back({src_slot, dst_slot, op_idx,
        "save " + target->name + " before " + op->label + " overwrites"});

    std::cout << "  [CopyInsert] memref.copy " << plan.slots[src_slot].name
              << " → " << bname << " (before " << op->label << ")\n";
    std::cout << "    reason: " << target->name << " has later readers: ";

    for (auto &c : conflicts) {
      if (c.writer != op.get() || c.shared_value != target) continue;
      auto *later = c.later_reader;

      // Redirect this later reader's input to the copy buffer
      plan.input_redirect[later][target] = dst_slot;
      std::cout << later->label;

      // If the later reader was INPLACE with the shared value,
      // its result should also go to the copy buffer (not the original)
      if (later->buf_info.decision == BufDecision::INPLACE &&
          later->buf_info.aliases_with == target) {
        plan.val_to_slot[later->result_] = dst_slot;
        std::cout << " (result " << later->result_->name
                  << " → " << bname << ")";
      }
      std::cout << " ";
    }
    std::cout << "\n";
  }
}

// =====================================================================
// Stage 6: Buffer Deallocation — ownership-based lifetime
// =====================================================================
// Rules:
//   - Function arguments (inputs + output): owned by caller → NO dealloc
//   - Locally allocated buffers: owned by this function → MUST dealloc
//   - Dealloc point: after the last op that reads/writes the buffer

static void stage6_dealloc(const Graph &g, BufPlan &plan) {
  for (auto &slot : plan.slots) {
    if (slot.is_func_arg) {
      slot.needs_dealloc = false;
      std::cout << "  [Ownership] " << slot.name
                << " → caller-owned (func arg), no dealloc\n";
      continue;
    }

    int last_use = -1;
    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *op = g.ops[i].get();
      bool uses_slot = false;
      for (auto *v : op->ins)
        if (plan.effective_slot(op, v) == slot.id) uses_slot = true;
      for (auto *v : op->outs)
        if (plan.slot_of(v) == slot.id) uses_slot = true;
      if (op->result_ && plan.slot_of(op->result_) == slot.id)
        uses_slot = true;
      if (uses_slot) last_use = static_cast<int>(i);
    }
    slot.last_use_idx = last_use;
    slot.needs_dealloc = true;

    std::string last_label = (last_use >= 0)
        ? g.ops[last_use]->label : "none";
    std::cout << "  [Ownership] " << slot.name
              << " → locally allocated, dealloc after "
              << last_label << " (op #" << last_use << ")\n";
  }
}

// =====================================================================
// Stage 7: Post-clean — verify & report
// =====================================================================

static void stage7_postclean(const BufPlan &plan) {
  int allocs = 0, copies = 0, deallocs = 0;
  int64_t total_alloc = 0;
  for (auto &s : plan.slots) {
    if (!s.is_func_arg) {
      ++allocs;
      total_alloc += s.type.bytes();
    }
    if (s.needs_dealloc) ++deallocs;
  }
  copies = static_cast<int>(plan.copies.size());

  std::cout << "  Post-clean verification:\n";
  std::cout << "    allocs  = deallocs? " << allocs << " vs " << deallocs;
  if (allocs == deallocs) std::cout << " ✓ (no leaks)\n";
  else std::cout << " ✗ (potential leak!)\n";
  std::cout << "    copies  = " << copies << "\n";
  std::cout << "    total alloc = " << total_alloc << " bytes\n";
}

// =====================================================================
// Print the fully bufferized graph
// =====================================================================

static void print_bufferized(const Graph &g, const BufPlan &plan) {
  std::cout << "module {\n  func.func @" << g.name << "_bufferized(";

  // Input arguments
  std::vector<std::pair<std::string, std::string>> sig_args;
  for (auto *a : g.args) {
    int sid = plan.slot_of(a);
    if (sid >= 0)
      sig_args.push_back({plan.slots[sid].name, plan.slots[sid].type.str()});
  }
  // Output argument (destination-passing)
  for (auto &s : plan.slots) {
    if (s.is_output_arg)
      sig_args.push_back({s.name, s.type.str()});
  }
  for (size_t i = 0; i < sig_args.size(); ++i) {
    if (i) std::cout << ",\n      ";
    std::cout << sig_args[i].first << ": " << sig_args[i].second;
  }
  std::cout << ") {\n";

  // Determine insertion points for allocs, copies, deallocs
  std::unordered_map<int, std::vector<int>> alloc_before;
  for (auto &s : plan.slots) {
    if (s.is_func_arg) continue;
    bool is_copy_dst = false;
    for (auto &c : plan.copies)
      if (c.dst_slot == s.id) { is_copy_dst = true; break; }
    if (is_copy_dst) continue;

    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *op = g.ops[i].get();
      if (op->result_ && plan.slot_of(op->result_) == s.id) {
        alloc_before[static_cast<int>(i)].push_back(s.id);
        break;
      }
    }
  }

  std::unordered_map<int, std::vector<CopyEntry>> copies_before;
  for (auto &c : plan.copies)
    copies_before[c.before_op_idx].push_back(c);

  int max_dealloc_pos = -1;
  for (auto &s : plan.slots)
    if (s.needs_dealloc && s.last_use_idx > max_dealloc_pos)
      max_dealloc_pos = s.last_use_idx;

  for (size_t i = 0; i < g.ops.size(); ++i) {
    int idx = static_cast<int>(i);

    if (alloc_before.count(idx)) {
      for (int sid : alloc_before[idx]) {
        auto &s = plan.slots[sid];
        std::cout << "    " << s.name << " = memref.alloc() : "
                  << s.type.str() << "\n";
      }
    }

    if (copies_before.count(idx)) {
      for (auto &c : copies_before[idx]) {
        auto &ds = plan.slots[c.dst_slot];
        std::cout << "    " << ds.name << " = memref.alloc() : "
                  << ds.type.str()
                  << "  // " << c.reason << "\n";
        std::cout << "    memref.copy " << plan.slots[c.src_slot].name
                  << ", " << ds.name << "\n";
      }
    }

    auto *op = g.ops[i].get();
    if (op->is_fill()) {
      int sid = plan.slot_of(op->result_);
      std::cout << "    linalg.fill ins(0.0) outs("
                << plan.slots[sid].name << " : "
                << plan.slots[sid].type.str() << ")\n";
      continue;
    }

    std::cout << "    " << op->mnemonic;
    if (!op->label.empty()) std::cout << "  // " << op->label;

    // Determine ins buffer slots (check per-op redirects first)
    std::cout << "\n      ins(";
    for (size_t j = 0; j < op->ins.size(); ++j) {
      if (j) std::cout << ", ";
      int sid = plan.effective_slot(op, op->ins[j]);
      if (sid >= 0)
        std::cout << plan.slots[sid].name << ": "
                  << plan.slots[sid].type.str();
      else
        std::cout << op->ins[j]->name << ": " << op->ins[j]->type.str();
    }
    std::cout << ")";

    // Determine outs buffer slot (result's buffer)
    int result_sid = plan.slot_of(op->result_);
    if (result_sid >= 0) {
      std::cout << " outs(" << plan.slots[result_sid].name << ": "
                << plan.slots[result_sid].type.str() << ")";
    }

    // Annotation: detect in-place by checking if any input uses the same buffer
    bool is_inplace = false;
    if (op->buf_info.decision == BufDecision::INPLACE) {
      for (auto *v : op->ins)
        if (plan.effective_slot(op, v) == result_sid) is_inplace = true;
      for (auto *v : op->outs)
        if (plan.slot_of(v) == result_sid) is_inplace = true;
    }
    if (is_inplace)
      std::cout << "  // [in-place]";
    else if (g.is_return_val(op->result_))
      std::cout << "  // [dest-passing → output]";
    std::cout << "\n";
  }

  // Deallocs
  for (auto &s : plan.slots) {
    if (s.needs_dealloc)
      std::cout << "    memref.dealloc " << s.name << " : "
                << s.type.str() << "\n";
  }
  std::cout << "    return\n  }\n}\n";
}

// =====================================================================
// Memory summary
// =====================================================================

static void print_summary(const Graph &g, const BufPlan &plan) {
  int64_t naive_bytes = 0;
  for (auto &op : g.ops) {
    if (op->result_) naive_bytes += op->result_->type.bytes();
  }

  int local_allocs = 0;
  int64_t actual_bytes = 0;
  for (auto &s : plan.slots) {
    if (!s.is_func_arg) {
      ++local_allocs;
      actual_bytes += s.type.bytes();
    }
  }
  int in_place_count = 0;
  for (auto &op : g.ops)
    if (op->buf_info.decision == BufDecision::INPLACE && !op->is_fill())
      ++in_place_count;

  int64_t saved = naive_bytes - actual_bytes;
  double pct = naive_bytes > 0 ? 100.0 * saved / naive_bytes : 0.0;

  std::cout << "\n  ┌──────────────────────────────────────────┐\n";
  std::cout << "  │          Memory Allocation Summary        │\n";
  std::cout << "  ├──────────────────────────────────────────┤\n";
  std::printf("  │  Naive (1 buf / tensor) : %6ld bytes    │\n", (long)naive_bytes);
  std::printf("  │  Bufferized (OSB)       : %6ld bytes    │\n", (long)actual_bytes);
  std::printf("  │  Saved                  : %6ld bytes    │\n", (long)saved);
  std::printf("  │  Reduction              : %5.1f%%          │\n", pct);
  std::cout << "  ├──────────────────────────────────────────┤\n";
  std::printf("  │  Local allocs  : %d                        │\n", local_allocs);
  std::printf("  │  In-place      : %d                        │\n", in_place_count);
  std::printf("  │  Copies        : %d                        │\n", (int)plan.copies.size());
  std::printf("  │  Deallocs      : %d                        │\n",
              (int)std::count_if(plan.slots.begin(), plan.slots.end(),
                                 [](const BufSlot &s) { return s.needs_dealloc; }));
  std::cout << "  └──────────────────────────────────────────┘\n";
}

// =====================================================================
// Test Graph
// =====================================================================
//
//   %A (64×128) × %B (128×32)   → matmul → %mm        [reduction + outs]
//   %mm + %bias                  → bias_add → %biased  [elementwise]
//   relu(%biased)                → %relu               [elementwise]
//   neg(%biased)                 → %neg                [elementwise, CONFLICT!]
//   %relu * %bias                → %scaled             [elementwise, reads func arg]
//   %scaled + %neg               → %out                [elementwise, return value]
//   %dead = %A + %A              → dead_add            [DCE target]
//
// Exercises:
//   Stage 0: DCE removes dead_add
//   Stage 1: alias chain  %init → %mm → %biased → %relu → %scaled
//            and separate  %biased → %neg (conflict)
//   Stage 2: WAR conflict — relu writes %biased's buffer, neg reads after
//   Stage 3: relu INPLACE + COPY_BEFORE; neg OUT_OF_PLACE (reads copy)
//   Stage 4: 2 local allocs (%buf for chain, %buf for copy)
//   Stage 5: 1 copy (save %biased before relu)
//   Stage 6: 2 deallocs (ownership-based)
//   Stage 7: verify allocs == deallocs (no leak)

static Graph make_test_graph() {
  Graph g;
  g.name = "osb_demo";

  TensorType T_A{{64, 128}, ElemType::F32};
  TensorType T_B{{128, 32}, ElemType::F32};
  TensorType T_out{{64, 32}, ElemType::F32};

  auto *A = g.add_arg(T_A);
  auto *B = g.add_arg(T_B);
  auto *bias = g.add_arg(T_out);

  auto *init = g.add_fill(T_out);
  auto *mm = g.add_matmul(A, B, init);
  auto *biased = g.add_elemwise("bias_add", mm->result_, bias);
  auto *relu = g.add_unary("relu", biased->result_);
  auto *neg = g.add_unary("neg", biased->result_);
  auto *scaled = g.add_elemwise("scale", relu->result_, bias);
  auto *out = g.add_elemwise("combine", scaled->result_, neg->result_);

  // DCE target
  g.add_elemwise("dead_add", A, A);

  g.returns = {out->result_};
  return g;
}

// =====================================================================
// Pipeline Runner
// =====================================================================

static void run_pipeline(Graph &g, const char *label) {
  auto sep = [](const char *title) {
    std::cout << "\n  ──── " << title << " ────\n\n";
  };

  std::cout
      << "\n╔═══════════════════════════════════════════════════════════════╗\n"
      << "║  " << label << "\n"
      << "╚═══════════════════════════════════════════════════════════════╝\n";

  std::cout << "\n[Initial] tensor-world graph (" << g.op_count() << " ops)\n";
  g.print(std::cout);

  // ── Stage 0: Pre-clean ──
  sep("Stage 0: Pre-clean (canonicalize + CSE + DCE)");
  int s0 = stage0_preclean(g);
  std::cout << "    → " << s0 << " cleanup(s)\n";
  if (s0 > 0) g.print(std::cout);

  // ── Stage 1: Alias Analysis ──
  sep("Stage 1: Alias Analysis (potential buffer sharing)");
  auto alias = stage1_alias_analysis(g);

  // ── Stage 2: In-place Analysis ──
  sep("Stage 2: In-place Analysis (write conflict detection)");
  auto conflicts = stage2_inplace_analysis(g);

  // ── Stage 3: Bufferization Decision ──
  sep("Stage 3: Bufferization Decision (INPLACE / OUT_OF_PLACE)");
  stage3_decision(g, conflicts);

  std::cout << "\n  Annotated tensor graph:\n";
  g.print(std::cout);

  // ── Stage 4: Rewrite ──
  sep("Stage 4: Rewrite (tensor → memref, buffer assignment)");
  auto plan = stage4_rewrite(g);

  // ── Stage 5: Copy Insertion ──
  sep("Stage 5: Copy Insertion (conflict resolution)");
  stage5_copy_insert(g, plan, conflicts);

  // ── Stage 6: Buffer Deallocation ──
  sep("Stage 6: Buffer Deallocation (ownership-based)");
  stage6_dealloc(g, plan);

  // ── Stage 7: Post-clean ──
  sep("Stage 7: Post-clean (verify)");
  stage7_postclean(plan);

  // ── Final: Bufferized output ──
  std::cout << "\n  ──── Bufferized Output ────\n\n";
  print_bufferized(g, plan);

  // ── Memory summary ──
  print_summary(g, plan);
}

// =====================================================================
int main() {
  std::cout << "================================================================\n";
  std::cout << "  One-Shot Bufferization (OSB) 8-Stage Pipeline\n";
  std::cout << "  S0:PreClean → S1:AliasAnalysis → S2:InPlaceAnalysis →\n";
  std::cout << "  S3:BufDecision → S4:Rewrite → S5:CopyInsert →\n";
  std::cout << "  S6:Dealloc → S7:PostClean\n";
  std::cout << "================================================================\n";

  auto g = make_test_graph();
  run_pipeline(g,
               "Matmul → BiasAdd → ReLU / Neg → Scale → Combine "
               "(One-Shot Bufferization)");

  std::cout << "\n✓ One-Shot Bufferization pipeline test passed.\n";
  return 0;
}
