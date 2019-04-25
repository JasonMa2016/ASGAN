import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model','-m',type=str)
args = parser.parse_args()
jeff = args.model
print(jeff)
np.savetxt(jeff+'/blah.npy',np.zeros(1))
