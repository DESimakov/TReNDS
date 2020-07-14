import argparse

import sys
import warnings

from age import run as age_run
from d11 import run as d11_run
from d12 import run as d12_run
from d21 import run as d21_run
from d22 import run as d22_run

# silence warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser(description='Train and inference')

    parser.add_argument('--seeds', default='all',
                        help='number of seeds to train. "All" coresponds to 7 seeds, everything else is 1 seed. Default "All"')


    args = parser.parse_args()
    seeds = [0]
    if args.seeds == 'all':
        seeds.extend([1, 2, 3, 4, 5, 20])

    # run for d12
    d12_run()

    # train and predict for all seeds
    for s in seeds:
    	age_run(s)
    	d11_run(s)
    	d21_run(s)
    	d22_run(s)

if __name__ == '__main__':
    main()
