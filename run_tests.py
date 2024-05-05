from stable_audio_tools import get_pretrained_model
from stable_audio_tools.interface.testing import runTests
print(runTests)  # Check if it prints a function reference


import torch

def main(args):
    torch.manual_seed(42)
    runTests(model_config_path = args.model_config, 
        ckpt_path=args.ckpt_path, 
        pretrained_name=args.pretrained_name, 
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        model_half=args.model_half,
        output_dir=args.output_dir,
        json_dir=args.json_dir
    )
    




if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Run generation tests')
    parser.add_argument('--pretrained-name', type=str, help='Name of pretrained model', required=False)
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)
    parser.add_argument('--output-dir', type=str, help='Path to output directory', required=True)
    parser.add_argument('--json-dir', type=str, help='Path to directory containing JSON files', required=True)
    print("Running tests")

    print("Arguments provided:", sys.argv[1:])
    
    args = parser.parse_args()
    print("Parsed arguments:", args)
    main(args)

    


