define i32 @test(i32 %x) {
entry:
%a = add i32 %x, 0
br label %B

B:
br label %C

C:
ret i32 %a
}
