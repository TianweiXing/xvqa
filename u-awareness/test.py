import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decay', type=float, default=0.999, help='Running average: decay rate for updating params, default 0.999. (If set to zero, no RA)')
    parser.add_argument("--load_embd", type=bool, default=False, help="If loading pre-trained GLOVE embedding, default is False")
    parser.add_argument("--out_name", type=str, default='try', help="output directory, default \'try\'")

    arguments = parser.parse_args()
    return arguments


args = parse_args()
print(args)