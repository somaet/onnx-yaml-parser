import os
import yaml
import onnx
import argparse
import onnxruntime as ort
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Parsing yaml to extract onnx model')

    parser.add_argument('--debug', action='store_true', help='Debugging mode')

    parser.add_argument(
        '--onnx_file',
        dest='onnx_file',
        help='The onnx model path.',
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the extract model.',
        type=str,
        default='./output'
    )
    return parser.parse_args()


def extract_model(onnx_path, save_dir_path):

    # 1번 분할 모델 
    input_names = ['data','resnetv10_stage1_conv1_fwd']
    output_names = ['resnetv10_stage1_conv0_fwd', 'resnetv10_dense0_fwd']

    # 2번 분할 모델 
    #input_names = ['resnetv10_stage1_conv0_fwd']
    #output_names = ['resnetv10_stage1_conv1_fwd'] 

    # 모델 추출 
    onnx.utils.extract_model(onnx_path, save_dir_path + 'model' + '.onnx',
                                input_names, output_names)




def main(args):
    # Extract ONNX model
    extract_model(args.onnx_file, args.save_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)

