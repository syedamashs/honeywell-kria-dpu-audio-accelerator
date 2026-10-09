import os
import glob

for p in glob.glob("models/**/*.xmodel", recursive=True):
    data = open(p, "rb").read()
    print("=" * 50)
    print("File:", p, len(data), "bytes")
    # Search for fingerprint or DPUCZDX8G
    idx = 0
    found = []
    for term in [b"DPUCZDX8G", b"B3136", b"B4096", b"fingerprint", b"ISA"]:
        pos = data.find(term)
        if pos != -1:
            snippet = data[max(0, pos-20):min(len(data), pos+60)]
            found.append((term.decode(), snippet))
    for term, snippet in found:
        print(f"  {term}: {repr(snippet)}")
