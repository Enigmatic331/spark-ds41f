"""Stop existing baseline containers by verified ID, keeping them restartable.

Operator tool for the baseline only. No image build/run, container removal, or
engine launch. The caller must independently verify recovery, service identity,
idle state and timers immediately before invoking --stop. AI-assisted local code.
"""
import argparse
import concurrent.futures
import re
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", action="append", required=True)
    p.add_argument("--ssh-config", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--stop", action="store_true")
    a = p.parse_args()
    if len(set(a.host)) != 4 or len(a.host) != 4:
        p.error("four distinct rank hosts required")
    if not all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", x) for x in [*a.host, a.container]):
        p.error("invalid host or container")

    def remote(host, *args):
        return subprocess.check_output(
            ["ssh", "-F", a.ssh_config, "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
             host, shlex.join(args)], text=True, timeout=60).strip()

    def identity(host, running, expected_id=None):
        data = remote(host, "docker", "inspect", "--type", "container", "--format",
                      "{{.Id}} {{.State.Running}} {{.HostConfig.AutoRemove}} {{.Name}} {{.Config.Image}}",
                      a.container).split()
        if (len(data) != 5 or not re.fullmatch(r"[0-9a-f]{64}", data[0])
                or data[1:] != [running, "false", "/" + a.container, a.image]
                or (expected_id is not None and data[0] != expected_id)):
            raise RuntimeError(f"{host}: baseline identity/state mismatch")
        return data[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        ids = dict(zip(a.host, pool.map(lambda host: identity(host, "true"), a.host)))
        if a.stop:
            def stop(host):
                remote(host, "docker", "stop", "-t", "30", ids[host])
                identity(host, "false", ids[host])
            list(pool.map(stop, a.host))
    print("all four baseline identities verified; " + (
        "original containers stopped and retained" if a.stop else "dry run, no change"))


if __name__ == "__main__":
    main()
