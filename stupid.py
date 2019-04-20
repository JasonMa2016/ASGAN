import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model','-m',type=str)
args = parser.parse_args()
jeff = args.model
print(jeff)
