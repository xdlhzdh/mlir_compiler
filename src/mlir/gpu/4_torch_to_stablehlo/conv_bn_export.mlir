module {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<i64>, %arg3: tensor<1x3x32x32xf32>) -> tensor<1x8x30x30xf32> {
    %cst = stablehlo.constant dense_resource<torch_tensor_8_torch.float32_2> : tensor<8xf32>
    %cst_0 = stablehlo.constant dense_resource<torch_tensor_8_torch.float32_1> : tensor<8xf32>
    %cst_1 = stablehlo.constant dense_resource<torch_tensor_8_3_3_3_torch.float32> : tensor<8x3x3x3xf32>
    %cst_2 = stablehlo.constant dense_resource<torch_tensor_8_torch.float32> : tensor<8xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<8xf32>
    %cst_4 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_5 = arith.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.convolution(%arg3, %cst_1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x32x32xf32>, tensor<8x3x3x3xf32>) -> tensor<1x8x30x30xf32>
    %1 = stablehlo.reshape %cst_2 : (tensor<8xf32>) -> tensor<8x1x1xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x8x30x30xf32>
    %4 = stablehlo.convert %cst_4 : (tensor<1xf64>) -> tensor<1xf32>
    %5 = stablehlo.reshape %4 : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %arg1, %6 : tensor<8xf32>
    %8 = stablehlo.sqrt %7 : tensor<8xf32>
    %9 = stablehlo.divide %cst_3, %8 : tensor<8xf32>
    %10 = stablehlo.convert %cst_5 : (tensor<1xi64>) -> tensor<1xf32>
    %11 = stablehlo.reshape %10 : (tensor<1xf32>) -> tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %13 = stablehlo.multiply %9, %12 : tensor<8xf32>
    %14 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<8x1xf32>
    %15 = stablehlo.reshape %14 : (tensor<8x1xf32>) -> tensor<8x1x1xf32>
    %16 = stablehlo.reshape %13 : (tensor<8xf32>) -> tensor<8x1xf32>
    %17 = stablehlo.reshape %16 : (tensor<8x1xf32>) -> tensor<8x1x1xf32>
    %18 = stablehlo.broadcast_in_dim %15, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %19 = stablehlo.subtract %3, %18 : tensor<1x8x30x30xf32>
    %20 = stablehlo.broadcast_in_dim %17, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<1x8x30x30xf32>
    %22 = stablehlo.reshape %cst_0 : (tensor<8xf32>) -> tensor<8x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<8x1xf32>) -> tensor<8x1x1xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %25 = stablehlo.multiply %21, %24 : tensor<1x8x30x30xf32>
    %26 = stablehlo.reshape %cst : (tensor<8xf32>) -> tensor<8x1xf32>
    %27 = stablehlo.reshape %26 : (tensor<8x1xf32>) -> tensor<8x1x1xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [1, 2, 3] : (tensor<8x1x1xf32>) -> tensor<1x8x30x30xf32>
    %29 = stablehlo.add %25, %28 : tensor<1x8x30x30xf32>
    return %29 : tensor<1x8x30x30xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_torch.float32_2: "0x040000000000000000000000000000000000000000000000000000000000000000000000",
      torch_tensor_8_torch.float32_1: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F",
      torch_tensor_8_3_3_3_torch.float32: "0x04000000AC122DBE553BAD3CDA5942BE793589BDCD46533D8550353E4B2AF7BD723B3ABDE962433DF22FF3BD60F737BD3F29E5BD89A71E3CFBD5253EEAEAC53D3A0E343DA376F73D0143A03D97C7A03D71880E3C8FB300BE8A5C3BBE3163FF3CDA0517BD98210F3E8BDCCBBD9B412E3E30C039BE66C1F1BC05AD44BE994486BD7D907BBDEC62ACBCAA07A33D9C7309BE6EF8AFBC235CBA3D622598BD0704AD3DFA35C93DA2E182BDF043343EE91928BE5BC10BBE40B5263CA0DE1B3D80C1FE3D8FD857BD8765823C760D173E494FCD3DBD73053B28F937BE132DA5BDB552303EBC122ABED68039BEF520373E5AC416BE2F44023D0D5422BEA3FD0D3E83A833BE7508EE3DD07BC9BDE10E0EBEC90D3A3E62963C3E41C52C3EAC9B02BEEDA0313EE22514BE472F26BE200207BE9537173E9A0E36BDCE94673D651B4B3994BACF3BD505FA3D83618DBBF395AE3DF6CF643D651993BDE7FF33BE82E0073E95A6033DE2BD85BD076D25BBE23507BC589A0ABE34B8B7BC939A01BE1C4B1EBDF91CB4BD08D17B3DBDDDB8BD67C81FBE53A60E3ED16727BE0AF636BED7AD3EBE4571583D2EFF9A3D0FD8B4BD4E58A73DFFBAB9BDD99FAE3A675C333E3ACB9F3DE3AB243EF8C1773DFDF8E23D19DF163C4B33D73CADBE8C3DB937B43CB543FC3B609E273E06B2013E8DBF073EEDD1C6BD956701BEB9CA36BD8203083ED1D8843D3F43333E0E1B343E864013BDA3900D3E45E326BDD0AF00BEB05B44BEE9A00DBE7000ACBD9C40EB3D7616D0BD2838783DC6D325BD4FDE953D0AD00F3E4F56793A12E2D3BDEBC1E33DD5DDA9BDCD9D253E0C31F43DE0BAFB3DED25D13C9B9B4DBDADC16FBD058525BECB9DE4BD35C83F3DD8062FBE6BEA063E062CC8BDA3953FBEFA44173ED334E83D33E5883DF2F023BE993DC5BDDA6CCD3D61F8C6BD9BE976BD773F06BE37823EBE7D0D273D5510F1BDCDFC833C0638B23D288FF13DAF61F7BD213B1BBE5304393DD474383D2E70E23DE4632C3E8DCBA43DF012853CDB3B8ABD495B2B3E177661BA1F1621BE3E94FFBD933818BEB1B807BE2FDAF73D1A85043EA7662CBD005DE63D4047343E63070FBED4F0D13DA7951D3EFAA6933C4C390E3E155BC53DB669873D2ADBBFBDE5A23F3CC249733CDADF4DBDB09FB83D8483053E7A8EC3BC186ED1BBBA1BCC3DB55CE43CF04CA5BCCFCA3C3E2796283E558EF9BDEFF7BB3C5FAC323E",
      torch_tensor_8_torch.float32: "0x04000000A4BEA2BDEE7BCC3DE4A6CABD6D94163C1BEE2BBAFF10E1BD05336FBC7619003E"
    }
  }
#-}
