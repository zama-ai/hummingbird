import onnx
from hummingbird.ml import convert
from hummingbird.ml import constants
import pandas as pd

m = "/root/hummingbird/tests/errorcode/mymodel.onnx"
# csvstr = "/root/hummingbird/tests/errorcode/X.csv"
# csv = pd.read_csv(csvstr)


csvstr_pruned = "/root/hummingbird/tests/errorcode/X_pruned.csv"
csv_20 = pd.read_csv(csvstr_pruned)


with open(m, "rb") as binary_file:
    modstr = binary_file.read()

mod = onnx.load_from_string(modstr)

# this will give an error
# hbout = convert(mod, "torch", csv)
# hbout.predict(csv)


hbout = convert(mod, "torch", csv_20)
print(hbout.predict(csv_20))
