"""
Export the best fine-tuned/merged model to ONNX for faster CPU/GPU inference
in production (no PyTorch/transformers Python overhead needed at serve time).

Usage:
  python src/export_onnx.py --model_dir results/peft__qlora__telugu/model \
      --base_model_for_adapter xlm-roberta-base --out_dir onnx_model

Requires: pip install optimum[onnxruntime]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--base_model_for_adapter", default=None)
    p.add_argument("--out_dir", default="onnx_model")
    p.add_argument("--quantize", action="store_true",
                    help="Also produce a dynamic-quantized int8 ONNX model for CPU speed")
    args = p.parse_args()

    from infer import HateSpeechClassifier

    # Load (and merge, if adapter) via our wrapper, then re-save as a plain
    # HF model so optimum can export it cleanly.
    clf = HateSpeechClassifier(args.model_dir, base_model_for_adapter=args.base_model_for_adapter)
    merged_dir = os.path.join(args.out_dir, "_merged_hf")
    os.makedirs(merged_dir, exist_ok=True)
    clf.model.save_pretrained(merged_dir)
    clf.tokenizer.save_pretrained(merged_dir)

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    ort_model = ORTModelForSequenceClassification.from_pretrained(merged_dir, export=True)
    ort_model.save_pretrained(args.out_dir)
    AutoTokenizer.from_pretrained(merged_dir).save_pretrained(args.out_dir)
    print(f"Exported ONNX model to {args.out_dir}")

    if args.quantize:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(ort_model)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quant_dir = args.out_dir + "_int8"
        quantizer.quantize(save_dir=quant_dir, quantization_config=qconfig)
        print(f"Exported int8-quantized ONNX model to {quant_dir}")


if __name__ == "__main__":
    main()
