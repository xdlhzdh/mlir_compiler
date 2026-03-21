import warnings

import onnxruntime as ort
import numpy as np
import onnx
import torch

from onnx import shape_inference


class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2 + 1


def _print_block(title: str, hint: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")
    if hint:
        print(f"提示: {hint}\n")


def main() -> None:
    model = Model()
    model.eval()
    x = torch.randn(1, 3)
    onnx_path = "../../../../build/model.onnx"

    _print_block(
        "1) 导出 ONNX",
        "使用新版 ONNX exporter 导出模型，并关闭 external data 以只生成一个 model.onnx 文件。",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        torch.onnx.export(model, x, onnx_path, opset_version=18, external_data=False)
    print(f"导出完成: {onnx_path}")

    _print_block(
        "2) 加载并打印 ONNX Graph",
        "先看图的文本表示，确认输入、输出和算子链是否符合预期。",
    )
    model_proto = onnx.load(onnx_path)
    print(onnx.printer.to_text(model_proto.graph))

    _print_block(
        "3) 遍历节点",
        "逐个打印 op_type、输入和输出，便于把图文本和节点结构对应起来。",
    )
    for node in model_proto.graph.node:
        print(node.op_type, list(node.input), list(node.output))

    _print_block(
        "4) Shape Inference 前",
        "查看中间值的 ValueInfo；很多模型在推断前这部分信息并不完整。",
    )
    print("Intermediate Value Info:", model_proto.graph.value_info)

    _print_block(
        "5) 执行 Shape Inference",
        "让 ONNX 根据算子语义补全更多中间张量的形状信息。",
    )
    model_proto = shape_inference.infer_shapes(model_proto)
    print("Shape inference complete.")

    _print_block(
        "6) Shape Inference 后",
        "再次查看 ValueInfo，观察中间张量的 shape 是否被补全。",
    )
    print("Intermediate Value Info:", model_proto.graph.value_info)
    onnx.save(model_proto, onnx_path)
    print(f"已保存带 shape 信息的模型: {onnx_path}")

    _print_block(
        "7) ONNX Runtime 推理",
        "加载模型并执行一次推理，验证导出的 ONNX 可以被 runtime 正常消费。",
    )
    session = ort.InferenceSession(onnx_path)
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected exactly 1 ONNX input, got {len(inputs)}")
    input_name = inputs[0].name
    input_data = np.random.randn(1, 3).astype(np.float32)
    print(f"Runtime input name: {input_name}")
    print(f"Runtime input data: {input_data}")
    output = session.run(None, {input_name: input_data})
    print("Runtime output:", output)


if __name__ == "__main__":
    main()
