# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..backend.ir_to_caffe import CaffeConverter
from ..backend.ir_to_caffe.caffe_converter import BackEnd
from ..converter_ir.ir_transform import IRTransform, TransformerRule
from ..frontend.mge_to_ir import MGE_FrontEnd


def mge_to_caffe(
    mge_fpath,
    prototxt="out.prototxt",
    caffemodel="out.caffemodel",
    outspec=None,
    use_empty_blobs=False,
    convert_backend: BackEnd = BackEnd.CAFFE,
):
    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    irgraph = MGE_FrontEnd(mge_fpath, outspec=outspec).resolve()

    transformer_options = [
        TransformerRule.EXPAND_MUL_ADD3,
        TransformerRule.FUSE_FOR_LEAKY_RELU,
        TransformerRule.CONV_ADD_ZERO_BIAS,
        TransformerRule.FUSE_CONV_BN,
    ]
    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)

    converter = CaffeConverter(
        transformed_irgraph, use_empty_blobs, convert_backend=convert_backend
    )
    converter.convert()

    assert isinstance(prototxt, str) and isinstance(
        caffemodel, str
    ), "'prototxt' and 'caffemodel' must be string"
    converter.dump(prototxt, caffemodel)
