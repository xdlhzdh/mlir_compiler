#include <iostream>
#include <set>
#include <string>

// --- NFA (非确定性有限状态自动机) ---
// NFA 状态定义
enum NFA_State { Q0, Q1, Q2 }; // Q2 为接受状态

/**
 * @brief NFA 状态转移函数：返回一个状态集合 (非确定性)
 * * NFA 的核心：对于 Q0 状态，输入 'a' 后，它同时保持 Q0（自环）并进入 Q1（匹配
 * 'a'）。
 */
std::set<NFA_State> NFA_Transition(const std::set<NFA_State> &current_states,
                                   char input_char) {
  std::set<NFA_State> next_states;

  for (NFA_State state : current_states) {
    if (state == Q0) {
      // 1. 匹配 (a|b)* 的自环 (保持在 Q0)
      next_states.insert(Q0);
      // 2. 尝试匹配 'ab' 的开头
      if (input_char == 'a') {
        next_states.insert(Q1); // 转移到 Q1
      }
    } else if (state == Q1) {
      // Q1 状态（已匹配 'a'）
      if (input_char == 'b') {
        next_states.insert(Q2); // 匹配成功 'b'，进入接受状态 Q2
      } else if (input_char == 'a') {
        next_states.insert(Q1); // 遇到 'a'，重新开始匹配
      }
      // 注意：NFA 的 Q0 自环已在上层循环中处理，这里不需要重复
    } else if (state == Q2) {
      // Q2 状态（已接受 'ab'），任何输入都保持 Q2 (匹配 (a|b)* 尾部)
      next_states.insert(Q2);
    }
  }

  // NFA 的非确定性体现：next_states 可能包含 Q0, Q1, Q2 的任何子集。
  return next_states;
}

void run_nfa(const std::string &input) {
  std::set<NFA_State> current_states = {Q0}; // 初始状态集合

  std::cout << "--- NFA 运行轨迹: " << input << " ---" << std::endl;
  for (char c : input) {
    current_states = NFA_Transition(current_states, c);

    std::cout << "  输入 '" << c << "': [";
    for (auto it = current_states.begin(); it != current_states.end(); ++it) {
      std::cout << "Q" << *it
                << (std::next(it) == current_states.end() ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    if (current_states.empty())
      break;
  }

  if (current_states.count(Q2)) {
    std::cout << "NFA 结果: 接受 (包含 'ab')" << std::endl;
  } else {
    std::cout << "NFA 结果: 拒绝" << std::endl;
  }
}

// --- DFA (确定性有限状态自动机) ---
// DFA 状态定义 (S0 对应 {Q0}, S1 对应 {Q0, Q1}, S2 对应 {Q0, Q2})
enum DFA_State { S0, S1, S2, S_ERROR };

/**
 * @brief DFA 状态转移函数：返回一个单一状态 (确定性)
 */
DFA_State DFA_Transition(DFA_State current_state, char input_char) {
  if (current_state == S_ERROR)
    return S_ERROR;

  // 使用 switch 结构实现查表逻辑
  switch (current_state) {
  case S0: // 对应 NFA 状态 {Q0}
    if (input_char == 'a')
      return S1; // S0 ('') -> S1 ('a')
    if (input_char == 'b')
      return S0; // S0 ('') -> S0 ('b')
    break;

  case S1: // 对应 NFA 状态 {Q0, Q1}
    if (input_char == 'a')
      return S1; // S1 ('a') -> S1 ('aa' 重新开始)
    if (input_char == 'b')
      return S2; // S1 ('a') -> S2 ('ab' 成功!)
    break;

  case S2: // 对应 NFA 状态 {Q0, Q2}
    if (input_char == 'a' || input_char == 'b')
      return S2;
    break; // 匹配 (a|b)* 的剩余部分

  default:
    return S_ERROR;
  }
  return S_ERROR; // 默认返回错误状态
}

void run_dfa(const std::string &input) {
  DFA_State current_state = S0; // 初始状态 (单一确定状态)

  std::cout << "\n--- DFA 运行轨迹: " << input << " ---" << std::endl;
  for (char c : input) {
    DFA_State next_state = DFA_Transition(current_state, c);

    std::cout << "  输入 '" << c << "': S" << current_state << " -> S"
              << next_state << std::endl;

    current_state = next_state;

    if (current_state == S_ERROR) {
      std::cout << "DFA 结果: 拒绝 (遇到无效字符)" << std::endl;
      return;
    }
  }

  if (current_state == S2) {
    std::cout << "DFA 结果: 接受 (包含 'ab')" << std::endl;
  } else {
    std::cout << "DFA 结果: 拒绝" << std::endl;
  }
}

int main() {
  std::string test_no_ab = "xba b a";

  // 运行 NFA
  run_nfa(test_no_ab);

  // 运行 DFA
  run_dfa(test_no_ab);

  std::string test_ab = "aababb";

  // 运行 NFA
  run_nfa(test_ab);

  // 运行 DFA
  run_dfa(test_ab);

  return 0;
}