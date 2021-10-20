import sys
import argparse

from dogsearch.model import Model


parser = argparse.ArgumentParser(prog="dogsearch.model", description="DogSearch Model")
parser.add_argument("type", type=str, help = "Model type")
parser.add_argument("image", type=str, help = "Path to the image")

args = parser.parse_args()

m = Model.create(args.type)
if m is None:
    sys.stderr.write("Uknown model type \"{}\"\n".format(args.type))
    sys.exit(1)

result = m.process(None, None)
print(result)
