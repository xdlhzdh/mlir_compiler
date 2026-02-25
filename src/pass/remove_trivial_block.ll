; 适用于 RemoveTrivialBlockPass 的测试 IR
; 包含多个 trivial block（仅含无条件跳转的中间块），运行 pass 后应被合并消除

define i32 @test(i32 %x) {
entry:
  br label %B1

; B1: trivial block (仅 br)
B1:
  br label %B2

; B2: trivial block (仅 br)
B2:
  br label %B3

; B3: trivial block (仅 br)
B3:
  br label %exit

exit:
  ret i32 %x
}
