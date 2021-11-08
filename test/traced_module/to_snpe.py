import json

import megengine as mge
from mgeconvert.converters.tm_to_caffe import tracedmodule_to_caffe


def save_quantize_params(qparams, path="quant_params.json"):
    bitwidth_dict = {
        "uint8": 8,
        "int8": 8,
        "int32": 32,
        "int16": 16,
    }
    quant_params = {}
    quant_params["activation_encodings"] = {}
    quant_params["param_encodings"] = {}
    for name, t in qparams.items():
        v_max, v_min = int(t["qmax"]), int(t["qmin"])

        if t["scale"] == "None":
            continue
        scale = float(t["scale"])
        zero_point = float(t["zero_point"]) if t["zero_point"] != "None" else 0
        param = {
            "min": float((v_min - zero_point) * scale),
            "max": float((v_max - zero_point) * scale),
            "scale": float(scale),
            "offset": int(zero_point),
            "bitwidth": bitwidth_dict[t["dtype"]],
        }
        if t["is_weight"]:
            quant_params["param_encodings"][name] = [param]
        else:
            quant_params["activation_encodings"][name] = [param]
    params = json.dumps(quant_params, indent=4)
    with open(path, "w") as f:
        f.write(params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="quant_params.json",
        help="input json file path",
    )
    parser.add_argument(
        "-t", "--trace_module", type=str, help="traced module file path"
    )
    parser.add_argument("--scale", type=float, default=16.0 / 128, help="input scale")
    parser.add_argument(
        "--zero_point", type=float, default=128, help="input zero_point"
    )
    parser.add_argument(
        "-o", "--output", default="snpe.json", type=str, help="output json file path"
    )
    args = parser.parse_args()

    if args.trace_module:
        net = mge.load(args.trace_module)
        tracedmodule_to_caffe(
            net,
            prototxt="caffe_module.txt",
            caffemodel="caffe_module.caffemodel",
            input_data_type="quint8",
            input_scales=args.scale,
            input_zero_points=args.zero_point,
            require_quantize=True,
            param_fake_quant=True,
            split_conv_relu=True,
            quantize_file_path=args.input,
        )

    with open(args.input, "rb") as f:
        qparams = json.load(f)
        save_quantize_params(qparams, args.output)
