"""Split a whole song into stems without transcribing, to look for a demo window.
Usage: separate_only.py <audio> <stems dir> [model, default htdemucs_6s]"""
import sys

from stemscribe.separate import separate

model = sys.argv[3] if len(sys.argv) > 3 else "htdemucs_6s"
for name, path in separate(sys.argv[1], sys.argv[2], model_name=model).items():
    print(name, path)
