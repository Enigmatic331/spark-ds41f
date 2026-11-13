"""Check the native launch flags and the installed vLLM max-output calculation."""
import argparse
import ast
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace


def check(root, launcher):
    lines=launcher.read_text().splitlines()
    def arg(flag):
        line,=(line.strip() for line in lines if line.strip().startswith(flag+" "))
        return json.loads(shlex.split(line)[1])
    assert arg("--default-chat-template-kwargs")=={"reasoning_effort":"low"}
    cfg=arg("--override-generation-config")
    cap=cfg["max_new_tokens"]
    assert cap==131072
    source=root/"vllm/entrypoints/serve/utils/api_utils.py"
    tree=ast.parse(source.read_text())
    func,=(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=="get_max_tokens")
    ns={"current_platform":SimpleNamespace(get_max_output_tokens=lambda _:sys.maxsize)}
    exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),"exec"),ns)
    maximum=ns["get_max_tokens"]
    for length,requested,want in [(40,None,cap),(40,cap+4000,cap),(40,17,17),(1048576-100,None,100)]:
        assert maximum(1048576,requested,length,{"max_tokens":cap},cap)==want
    print("PASS: output budget selection for omitted, oversized, smaller and near-context requests")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("vllm_root",type=Path);p.add_argument("launcher",type=Path)
    args=p.parse_args();check(args.vllm_root,args.launcher)
