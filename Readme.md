# Usage
---
python MHsampler.py  --data_path [data_path] --model_path [model_path] 

---
Some optional args:
[--init] whether start from scratch
[--ExtVector] whether use extern precomputed vectors; if use, the filename should be the same as data_path with the suffix ".npz"
[--VectorDim] dimension of extern vectors
[--Faiss] whether use Faiss to accelerate
[--Df] whether dynamically compute the feature vectors after composing
---
More information about the args please see the code MHsampler.py.