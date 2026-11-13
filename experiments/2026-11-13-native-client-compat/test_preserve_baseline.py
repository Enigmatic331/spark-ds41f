"""Offline controller checks; no subprocess, SSH or container engine is run."""
import contextlib
import io
import shlex
import unittest
from unittest.mock import patch
import preserve_baseline as target


class Preservation(unittest.TestCase):
    def simulate(self, *, stop=False, removable=False, replaced=False):
        hosts = ["rank%d" % n for n in range(4)]
        ids = {h: ("%064x" % (i + 1)) for i,h in enumerate(hosts)}
        stopped = set()
        self.stopped = stopped
        def fake(command, **unused):
            host, remote = command[-2:]
            cmd = shlex.split(remote)
            if cmd[:2] == ["docker", "inspect"]:
                ident = "a"*64 if replaced and host in stopped else ids[host]
                auto = "true" if removable and host == "rank3" else "false"
                return f"{ident} {'false' if host in stopped else 'true'} {auto} /baseline test:image\n"
            self.assertEqual(cmd, ["docker","stop","-t","30",ids[host]])
            stopped.add(host)
            return ids[host]
        argv = ["tool","--ssh-config","unused","--container","baseline","--image","test:image"]
        for host in hosts:
            argv.extend(["--host",host])
        if stop:
            argv.append("--stop")
        with patch("sys.argv",argv), patch.object(target.subprocess,"check_output",side_effect=fake), contextlib.redirect_stdout(io.StringIO()):
            target.main()

    def test_dry(self):
        self.simulate()
        self.assertFalse(self.stopped)

    def test_retain_all(self):
        self.simulate(stop=True)
        self.assertEqual(len(self.stopped),4)

    def test_fail_before_first_stop(self):
        with self.assertRaises(RuntimeError):
            self.simulate(stop=True,removable=True)
        self.assertFalse(self.stopped)

    def test_replacement_cannot_pass(self):
        with self.assertRaises(RuntimeError):
            self.simulate(stop=True,replaced=True)


if __name__ == "__main__":
    unittest.main()
