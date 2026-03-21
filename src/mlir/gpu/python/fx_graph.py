"""
演示 torch.fx：symbolic_trace 得到 GraphModule、遍历节点、图级改写后 recompile。

运行: python fx_graph.py
"""

from __future__ import annotations

import operator

import torch
from torch import fx


def _print_block(title: str, hint: str) -> None:
    """统一打印区块标题与说明，便于区分不同输出。"""
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")
    if hint:
        print(f"提示: {hint}\n")


class MyModule(torch.nn.Module):
    """简单算子链: x -> x+1 -> *2，便于观察图结构。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 0
        z =  y * 2 * 1 + 0
        return z


def main() -> None:
    model = MyModule()
    traced = fx.symbolic_trace(model)

    _print_block(
        "1) FX Graph（IR 的文本表示）",
        "每个节点对应一次计算；边由 def/use 隐含在 args 里。",
    )
    print(traced.graph)

    _print_block(
        "2) 节点列表（op / target）",
        "op: placeholder / get_attr / call_function / call_method / output 等；"
        "target 对 call_* 为具体函数或方法。",
    )
    for node in traced.graph.nodes:
        print(f"  {node.op:16} {node.target}")

    # 示例：删除恒等加法 add(x, 0) -> x（本图为 x+1，故通常 changed=0）
    _print_block(
        "3) 图改写（示例）",
        "遍历 call_function 且 target 为 torch.add 或 operator.add（`x+1` 常是后者）；"
        "若第二实参为字面量 0，则替换为第一个实参并擦除该节点。",
    )
    changed = 0
    for node in list(traced.graph.nodes):
        if node.op == "call_function" and node.target in (torch.add, operator.add):
            if len(node.args) >= 2 and node.args[1] == 0:
                node.replace_all_uses_with(node.args[0])
                traced.graph.erase_node(node)
                changed += 1
        if node.op == "call_function" and node.target in (torch.mul, operator.mul):
            if len(node.args) >= 2 and node.args[1] == 1:
                node.replace_all_uses_with(node.args[0])
                traced.graph.erase_node(node)
                changed += 1
    print(f"  本次改写移除的恒等 add 节点数: {changed}")

    traced.recompile()

    _print_block(
        "4) 改写后的 Python 代码（GraphModule.forward）",
        "recompile() 根据当前 graph 重新生成可执行的 forward；与 StableHLO 导出是不同层，但都基于同一套 FX 图思想。",
    )
    print(traced.code)


if __name__ == "__main__":
    main()
