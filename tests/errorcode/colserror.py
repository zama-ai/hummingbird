import onnx
from hummingbird.ml import convert
from hummingbird.ml import constants
import pandas as pd

m = "/root/hummingbird/tests/ravenerror/mymodel.onnx"
csvstr = "/root/hummingbird/tests/ravenerror/X.csv"
csv = pd.read_csv(csvstr)

with open(m, "rb") as binary_file:
    modstr = binary_file.read()

hbout = convert(onnx.load_from_string(modstr), "torch", csv)

# this will give an error
hbout.predict(csv)
