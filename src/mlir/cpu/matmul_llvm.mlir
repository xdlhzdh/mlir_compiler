module attributes {torch.debug_module_name = "MatmulModule"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @forward(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg10, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg12, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg11, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg13, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg0, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg1, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg2, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg3, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg5, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg4, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg6, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.constant(4 : index) : i64
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %20 = llvm.mlir.constant(4 : index) : i64
    %21 = llvm.mlir.constant(4 : index) : i64
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.mlir.constant(16 : index) : i64
    %24 = llvm.mlir.zero : !llvm.ptr
    %25 = llvm.getelementptr %24[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.mlir.constant(64 : index) : i64
    %28 = llvm.add %26, %27 : i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.sub %27, %31 : i64
    %33 = llvm.add %30, %32 : i64
    %34 = llvm.urem %33, %27 : i64
    %35 = llvm.sub %33, %34 : i64
    %36 = llvm.inttoptr %35 : i64 to !llvm.ptr
    %37 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.insertvalue %29, %37[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %36, %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(0 : index) : i64
    %41 = llvm.insertvalue %40, %39[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.insertvalue %20, %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %21, %42[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %21, %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.insertvalue %22, %44[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%18 : i64)
  ^bb1(%46: i64):  // 2 preds: ^bb0, ^bb5
    %47 = llvm.icmp "slt" %46, %17 : i64
    llvm.cond_br %47, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%18 : i64)
  ^bb3(%48: i64):  // 2 preds: ^bb2, ^bb4
    %49 = llvm.icmp "slt" %48, %17 : i64
    llvm.cond_br %49, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %50 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.mlir.constant(4 : index) : i64
    %52 = llvm.mul %46, %51 overflow<nsw, nuw> : i64
    %53 = llvm.add %52, %48 overflow<nsw, nuw> : i64
    %54 = llvm.getelementptr inbounds|nuw %50[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %54 : f32, !llvm.ptr
    %55 = llvm.add %48, %16 : i64
    llvm.br ^bb3(%55 : i64)
  ^bb5:  // pred: ^bb3
    %56 = llvm.add %46, %16 : i64
    llvm.br ^bb1(%56 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%18 : i64)
  ^bb7(%57: i64):  // 2 preds: ^bb6, ^bb14
    %58 = llvm.icmp "slt" %57, %17 : i64
    llvm.cond_br %58, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%18 : i64)
  ^bb9(%59: i64):  // 2 preds: ^bb8, ^bb13
    %60 = llvm.icmp "slt" %59, %17 : i64
    llvm.cond_br %60, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%18 : i64)
  ^bb11(%61: i64):  // 2 preds: ^bb10, ^bb12
    %62 = llvm.icmp "slt" %61, %17 : i64
    llvm.cond_br %62, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %63 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.getelementptr %63[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %66 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mul %57, %66 overflow<nsw, nuw> : i64
    %68 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.mul %61, %68 overflow<nsw, nuw> : i64
    %70 = llvm.add %67, %69 overflow<nsw, nuw> : i64
    %71 = llvm.getelementptr inbounds|nuw %65[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %72 = llvm.load %71 : !llvm.ptr -> f32
    %73 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.getelementptr %73[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.mul %61, %76 overflow<nsw, nuw> : i64
    %78 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mul %59, %78 overflow<nsw, nuw> : i64
    %80 = llvm.add %77, %79 overflow<nsw, nuw> : i64
    %81 = llvm.getelementptr inbounds|nuw %75[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %82 = llvm.load %81 : !llvm.ptr -> f32
    %83 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.mlir.constant(4 : index) : i64
    %85 = llvm.mul %57, %84 overflow<nsw, nuw> : i64
    %86 = llvm.add %85, %59 overflow<nsw, nuw> : i64
    %87 = llvm.getelementptr inbounds|nuw %83[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %88 = llvm.load %87 : !llvm.ptr -> f32
    %89 = llvm.fmul %72, %82 : f32
    %90 = llvm.fadd %88, %89 : f32
    %91 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.mlir.constant(4 : index) : i64
    %93 = llvm.mul %57, %92 overflow<nsw, nuw> : i64
    %94 = llvm.add %93, %59 overflow<nsw, nuw> : i64
    %95 = llvm.getelementptr inbounds|nuw %91[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %90, %95 : f32, !llvm.ptr
    %96 = llvm.add %61, %16 : i64
    llvm.br ^bb11(%96 : i64)
  ^bb13:  // pred: ^bb11
    %97 = llvm.add %59, %16 : i64
    llvm.br ^bb9(%97 : i64)
  ^bb14:  // pred: ^bb9
    %98 = llvm.add %57, %16 : i64
    llvm.br ^bb7(%98 : i64)
  ^bb15:  // pred: ^bb7
    llvm.return %45 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}

