; 适用于 RemoveTrivialLoopPass 的测试 IR
; 包含死循环（无 exit 的 loop），运行 pass 后应被删除，控制流重定向到 unreachable

define i32 @test() {
entry:
  br label %loop

; 死循环：仅跳回自身，无 exit
loop:
  br label %loop
}

; 条件进入死循环：entry 还有另一条可达路径到 normal_exit
; pass 不应把整个 entry 改成 unreachable（否则会误删正常路径）
define i32 @test_conditional_entry(i1 %cond) {
entry:
  br i1 %cond, label %loop, label %normal_exit

loop:
  br label %loop

normal_exit:
  ret i32 0
}
