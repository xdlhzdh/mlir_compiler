; 适用于 RemoveZeroTripLoopPass 的测试 IR
; 包含 trip count = 0 的循环（i < 0 恒假，循环体永不执行）
; 运行 pass 后应删除 loop 块，entry 直接跳到 exit

define i32 @test() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %cmp = icmp slt i32 %i, 0
  br i1 %cmp, label %body, label %exit

body:
  %i.next = add i32 %i, 1
  br label %loop

exit:
  ret i32 %i
}
