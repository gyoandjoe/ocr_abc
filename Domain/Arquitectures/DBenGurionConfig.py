
distributionUniformParams = {
    "conv1LowValue":-0.1,
    "conv1HighValue":0.1,

    "conv2LowValue":-0.1,
    "conv2HighValue":0.1,

    "conv3LowValue":-0.1,
    "conv3HighValue":0.1,

    "conv4LowValue":-0.1,
    "conv4HighValue":0.1,

    "conv5LowValue":-0.01,
    "conv5HighValue":0.01,

    "conv6LowValue":-0.01,
    "conv6HighValue":0.01,

    "fc1LowValue":-0.0001,
    "fc1HighValue":0.0001,

    "fc2LowValue":-0.0001,
    "fc2HighValue":0.0001,

    "SoftMLowValue":-0.00001,
    "SoftMHighValue":0.00001,

    "FC1BiasInit":1,
    "FC2BiasInit":1,
    "SoftMBiasInit":1
    }

distributionNormalDistParams = {
    "conv1InitMean":0,
    "conv1InitSD":0.1,

    "conv2InitMean":0,
    "conv2InitSD":0.1,

    "conv3InitMean":0,
    "conv3InitSD":0.1,

    "conv4InitMean":0,
    "conv4InitSD":0.1,

    "conv5InitMean":0,
    "conv5InitSD":0.01,

    "conv6InitMean":0,
    "conv6InitSD":0.01,

    "fc1InitMean":0,
    "fc1InitSD":0.001,

    "fc2InitMean":0,
    "fc2InitSD":0.001,

    "SoftMInitMean":0,
    "SoftMInitSD":0.0001,

    "FC1BiasInit":1,
    "FC2BiasInit":1,
    "SoftMBiasInit":1
}

layers_metaData = {
    'Conv1_NoFiltersOut': 128,
    'Conv1_NoFiltersIn': 1,
    'Conv1_sizeKernelW': 3,
    'Conv1_sizeKernelH': 3,
    'Conv1_sizeImgInH': 64,
    'Conv1_sizeImgInW': 64,

    'Conv2_NoFiltersOut': 128,
    'Conv2_NoFiltersIn': 128,
    'Conv2_sizeKernelW': 3,
    'Conv2_sizeKernelH': 3,
    'Conv2_sizeImgInH': 64,
    'Conv2_sizeImgInW': 64,
    # Pool 1
    'Conv3_NoFiltersOut': 256,
    'Conv3_NoFiltersIn': 128,
    'Conv3_sizeKernelW': 3,
    'Conv3_sizeKernelH': 3,
    'Conv3_sizeImgInH': 32,
    'Conv3_sizeImgInW': 32,

    'Conv4_NoFiltersOut': 256,
    'Conv4_NoFiltersIn': 256,
    'Conv4_sizeKernelW': 3,
    'Conv4_sizeKernelH': 3,
    'Conv4_sizeImgInH': 32,
    'Conv4_sizeImgInW': 32,
    # pool 2
    'Conv5_NoFiltersOut': 512,
    'Conv5_NoFiltersIn': 256,
    'Conv5_sizeKernelW': 3,
    'Conv5_sizeKernelH': 3,
    'Conv5_sizeImgInH': 16,
    'Conv5_sizeImgInW': 16,

    'Conv6_NoFiltersOut': 512,
    'Conv6_NoFiltersIn': 512,
    'Conv6_sizeKernelW': 3,
    'Conv6_sizeKernelH': 3,
    'Conv6_sizeImgInH': 16,
    'Conv6_sizeImgInW': 16,
    # pool 3
    'FC1_NoFiltersOut': 2048,
    'FC1_NoFiltersIn': 512,
    'FC1_sizeKernelW': 3,
    'FC1_sizeKernelH': 3,
    'FC1_sizeImgInH': 8,
    'FC1_sizeImgInW': 8,

    'FC2_NoFiltersIn': 2048, #512 * 8 * 8
    'FC2_NoFiltersOut': 2048,

    'SoftM_NoFiltersIn':2048,
    'SoftM_NoFiltersOut':62,



    'DO1_size_in': 2048,

    'DO2_size_in': 2048,
}

