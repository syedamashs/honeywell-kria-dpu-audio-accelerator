import os

data = open("models/compiled/dscnn_medium.xmodel", "rb").read()
print("Length:", len(data))

# Find string patterns
for s in [b"input", b"output", b"fix_point", b"shape", b"dims", b"kernel", b"subgraph"]:
    count = data.count(s)
    print(f"Count of {s}: {count}")

# Look at meta.json
if os.path.exists("models/compiled/meta.json"):
    print("meta.json:", open("models/compiled/meta.json").read())
