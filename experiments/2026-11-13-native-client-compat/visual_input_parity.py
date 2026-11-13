"""Compare the reviewed native V4.1 processor and checkpoint reference on CPU."""
import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("fixture-dir","reference","native","config"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    import torch
    import numpy as np
    from PIL import Image,ImageOps
    torch.set_num_threads(1)
    def load(path,sha,reference):
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha: raise ValueError("processor hash mismatch")
        names={"llm_grid","num_image_tokens","solve_resize_ratio","safe_resize","load_image","image_token_types"}
        if reference: names.add("plan_image_grid")
        defs=[node for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef) and node.name in names]
        if len(defs)!=len(names): raise ValueError("processor functions unavailable")
        # Only the audited pixel/grid functions; the input adapter accepts fixture bytes only.
        env=dict(torch=torch,np=np,math=math,Image=Image,ImageOps=ImageOps,io=io,
                 IMAGE_START=0,IMAGE=1,IMAGE_NEW_LINE=2,IMAGE_END=3,load_image_bytes=lambda rec:rec["data"])
        exec(compile(ast.Module(body=defs,type_ignores=[]),str(path),"exec"),env)
        return env
    ref=load(a.reference,"482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272",True)
    native=load(a.native,"91f66f0c84f55709c10f16127b486a8af0e5f291efe08b46d617830eed3e8cab",False)
    cfg=json.loads(a.config.read_text())["vision_config"]
    args=dict(patch_size=cfg["patch_size"],downsample_ratio=cfg["downsample_ratio"],
              max_n_token=cfg["max_image_tokens"],min_pixels=cfg["min_pixels"],max_wh_ratio=cfg["max_wh_ratio"])
    ref_args=SimpleNamespace(**{"vision_"+k:v for k,v in args.items()})
    rows=[]
    for name in ("etherscan-baseline.png","etherscan-candidate-clear.png","etherscan-debug-clear.png"):
        raw=(a.fixture_dir/name).read_bytes()
        r,*rg=ref["load_image"]({"data":raw},ref_args)
        with Image.open(io.BytesIO(raw)) as im: n,*ng=native["load_image"](im,**args)
        rtypes=ref["image_token_types"](*rg[-2:]);ntypes=native["image_token_types"](*ng[-2:])
        row=dict(fixture=name,vit_and_llm_grids=rg,patch_sha256=hashlib.sha256(r.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                 span_tokens=len(rtypes),exact_patches=torch.equal(r,n),exact_grid=rg==ng,exact_roles=torch.equal(rtypes,ntypes))
        rows.append(row)
    print(json.dumps(rows))
    if not all(row["exact_patches"] and row["exact_grid"] and row["exact_roles"] for row in rows): raise SystemExit(2)


if __name__=="__main__":
    main()
