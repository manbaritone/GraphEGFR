import argparse
from graphegfr.configs import Configs
from graphegfr.main import run

def parse_args():
    global DEBUG
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, dest="configs", help='enter configuration file path')
    parser.add_argument('--debug', default=False, action="store_true", dest='debug', help='debug file/test run')
    args = parser.parse_args()
    configs = Configs.parse(args.configs)
    DEBUG = args.debug
    return configs

if __name__ == "__main__":
    configs = parse_args()
    run(configs.to_dict(), DEBUG)