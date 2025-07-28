from math_verify import parse,verify
import json
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    res=[]
    with open(args.output_path, "r") as f:
        for line in f:
            res.append(json.loads(line))
    count = 1e-6
    probs = []
    for i in tqdm(range(len(res))):
        a1=parse("\\boxed{" + res[i]["answer"] + "}", parsing_timeout=5)
        a2=parse(res[i]["output"], parsing_timeout=5)
        # print(a1,a2)
        # break
        if verify(a1,a2):
            count += 1
    print(count / len(res))
