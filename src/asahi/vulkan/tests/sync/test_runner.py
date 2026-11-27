#!/usr/bin/env python3
"""Unit tests for the Borealis synchronization regression runner."""

import contextlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner


# Opening a FIFO blocks until the other end is opened too, so "block" never
# returns and "write"/"read" complete only when they run at the same time.
FAKE_TEST = r'''#!{python}
import os
import sys

operation = sys.argv[1]
if operation == "pass":
    sys.exit(0)
if operation == "fail":
    print("deliberate failure", file=sys.stderr)
    sys.exit(3)
if operation == "skip":
    sys.exit(77)
if operation == "block":
    os.open(os.environ["FIFO"], os.O_RDONLY)
    sys.exit(0)
if operation == "write":
    os.close(os.open(os.environ["FIFO"], os.O_WRONLY))
    sys.exit(0)
if operation == "read":
    os.close(os.open(os.environ["FIFO"], os.O_RDONLY))
    sys.exit(0)
if operation == "orphan":
    # A descendant in its own session keeps the output pipes open.
    import subprocess
    subprocess.Popen([sys.executable, sys.argv[0], "block"],
                     start_new_session=True)
    os.open(os.environ["FIFO"], os.O_RDONLY)
    sys.exit(0)
raise SystemExit(5)
'''


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bindir = self.root / "bin"
        self.bindir.mkdir()
        fake = self.bindir / "fake-test"
        fake.write_text(FAKE_TEST.format(python=sys.executable), encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, entries, timeout=1):
        path = self.root / "cases.json"
        path.write_text(
            json.dumps(
                {"version": 1, "defaults": {"timeout": timeout}, "cases": entries}
            ),
            encoding="utf-8",
        )
        return runner.load_cases(path)[1], path

    @staticmethod
    def leaf(name, operation, **extra):
        case = {
            "name": name,
            "tier": "strict",
            "executable": "fake-test",
            "args": [operation],
        }
        case.update(extra)
        return case

    def test_invalid_executable_path_is_rejected(self):
        with self.assertRaisesRegex(runner.HarnessError, "relative to --bindir"):
            self.load(
                [
                    {
                        "name": "escape",
                        "executable": "../fake-test",
                    }
                ]
            )

    def test_pass_fail_skip_xfail_and_xpass(self):
        cases, _ = self.load(
            [
                self.leaf("pass", "pass"),
                self.leaf("fail", "fail"),
                self.leaf("skip", "skip"),
                self.leaf("xpass", "pass"),
            ]
        )
        results, exit_code = runner.execute_suite(
            cases,
            self.bindir,
            xfails={"fail": "known", "xpass": "should still fail"},
            seed=12,
        )
        by_name = {result["name"]: result for result in results}
        self.assertEqual(by_name["pass"]["status"], "PASS")
        self.assertEqual(by_name["fail"]["status"], "XFAIL")
        self.assertEqual(by_name["skip"]["status"], "SKIP")
        self.assertEqual(by_name["xpass"]["status"], "XPASS")
        self.assertEqual(exit_code, 1)

    def test_all_skipped_returns_77(self):
        cases, _ = self.load(
            [
                self.leaf("skip", "skip"),
                {
                    "name": "optional-missing",
                    "tier": "strict",
                    "executable": "not-built",
                    "optional": True,
                },
            ]
        )
        results, exit_code = runner.execute_suite(cases, self.bindir)
        self.assertTrue(all(result["status"] == "SKIP" for result in results))
        self.assertEqual(exit_code, 77)

    def test_missing_strict_executable_is_harness_error(self):
        cases, _ = self.load(
            [{"name": "missing", "tier": "strict", "executable": "not-built"}]
        )
        results, exit_code = runner.execute_suite(cases, self.bindir)
        self.assertEqual(results[0]["status"], "HARNESS_ERROR")
        self.assertEqual(exit_code, 2)

    def fifo(self):
        path = self.root / "fifo"
        os.mkfifo(path)
        return {"env": {"FIFO": str(path)}}

    def test_timeout_is_test_failure(self):
        cases, _ = self.load([self.leaf("slow", "block", **self.fifo())])
        results, exit_code = runner.execute_suite(
            cases, self.bindir, timeout_override=0.2
        )
        self.assertEqual(results[0]["status"], "FAIL")
        self.assertTrue(results[0]["timed_out"])
        self.assertEqual(exit_code, 1)

    def test_timeout_with_output_held_open_is_bounded(self):
        fifo = self.fifo()
        cases, _ = self.load([self.leaf("orphan", "orphan", **fifo)])
        drain = runner.KILL_DRAIN_SECONDS
        runner.KILL_DRAIN_SECONDS = 0.2
        try:
            results, exit_code = runner.execute_suite(
                cases, self.bindir, timeout_override=0.2
            )
        finally:
            runner.KILL_DRAIN_SECONDS = drain
            # Release the descendant blocked opening the FIFO.
            os.close(os.open(fifo["env"]["FIFO"], os.O_WRONLY))
        self.assertEqual(results[0]["status"], "FAIL")
        self.assertTrue(results[0]["timed_out"])
        self.assertIn("kept its output open", results[0]["message"])
        self.assertLess(results[0]["duration_ms"], 5000)
        self.assertEqual(exit_code, 1)

    def test_non_finite_timeouts_are_rejected(self):
        for value in (float("nan"), float("inf"), 0, -1, True):
            with self.assertRaises(runner.HarnessError):
                self.load([self.leaf("pass", "pass", timeout=value)])
        cases, _ = self.load([self.leaf("pass", "pass")])
        with self.assertRaisesRegex(runner.HarnessError, "finite"):
            runner.execute_suite(
                cases, self.bindir, timeout_override=float("nan")
            )

    def test_compat_is_opt_in_but_explicit_selection_runs_it(self):
        strict = self.leaf("strict", "pass")
        compat = self.leaf("compat", "pass", tier="compat")
        cases, _ = self.load([strict, compat])
        default, _ = runner.execute_suite(cases, self.bindir)
        self.assertEqual([result["name"] for result in default], ["strict"])
        selected, _ = runner.execute_suite(
            cases, self.bindir, requested=["compat"]
        )
        self.assertEqual([result["name"] for result in selected], ["compat"])

    def test_opt_in_case_is_excluded_from_default_and_compat(self):
        strict = self.leaf("strict", "pass")
        compat = self.leaf("compat", "pass", tier="compat")
        opt_in = self.leaf("known-device-loss", "fail", opt_in=True)
        cases, _ = self.load([strict, compat, opt_in])
        default, _ = runner.execute_suite(cases, self.bindir)
        self.assertEqual([result["name"] for result in default], ["strict"])
        with_compat, _ = runner.execute_suite(
            cases, self.bindir, include_compat=True
        )
        self.assertEqual(
            [result["name"] for result in with_compat], ["strict", "compat"]
        )
        selected, exit_code = runner.execute_suite(
            cases, self.bindir, requested=["known-device-loss"]
        )
        self.assertEqual(
            [result["name"] for result in selected], ["known-device-loss"]
        )
        self.assertEqual(exit_code, 1)

    def test_opt_in_validation_and_parallel_guard(self):
        with self.assertRaisesRegex(runner.HarnessError, "opt_in must be boolean"):
            self.load([self.leaf("bad", "pass", opt_in="yes")])
        with self.assertRaisesRegex(runner.HarnessError, "cannot contain opt-in"):
            self.load(
                [
                    self.leaf("strict", "pass"),
                    self.leaf("optin", "fail", opt_in=True),
                    {
                        "name": "pressure",
                        "tier": "strict",
                        "members": ["strict", "optin"],
                    },
                ]
            )

    def test_suite_cases_json_excludes_dup_from_default(self):
        cases_path = Path(__file__).with_name("cases.json")
        _, cases = runner.load_cases(cases_path)
        default_names = [
            case["name"] for case in runner.select_cases(cases, [], False)
        ]
        compat_names = [
            case["name"] for case in runner.select_cases(cases, [], True)
        ]
        self.assertIn("submit-pack-ordering", default_names)
        self.assertNotIn("dup", cases["submit-pack-ordering"].get("args", []))
        self.assertIn("submit-pack-dup", cases)
        self.assertTrue(cases["submit-pack-dup"]["opt_in"])
        self.assertNotIn("submit-pack-dup", default_names)
        self.assertNotIn("submit-pack-dup", compat_names)
        self.assertEqual(
            [c["name"] for c in runner.select_cases(cases, ["submit-pack-dup"], False)],
            ["submit-pack-dup"],
        )

    def test_seed_makes_order_repeatable(self):
        cases, _ = self.load(
            [self.leaf(name, "pass") for name in ("one", "two", "three", "four")]
        )

        def order(seed):
            results, _ = runner.execute_suite(
                cases, self.bindir, repetitions=3, seed=seed
            )
            return [(result["iteration"], result["name"]) for result in results]

        self.assertEqual(order(9981), order(9981))
        self.assertEqual(len(order(9981)), 12)

    def test_parallel_case_starts_members_concurrently(self):
        common = self.fifo()
        cases, _ = self.load(
            [
                self.leaf("left", "write", **common),
                self.leaf("right", "read", **common),
                {
                    "name": "pressure",
                    "tier": "strict",
                    "members": ["left", "right"],
                },
            ]
        )
        results, exit_code = runner.execute_suite(
            cases, self.bindir, requested=["pressure"], seed=4
        )
        self.assertEqual(results[0]["status"], "PASS")
        self.assertEqual(
            [member["status"] for member in results[0]["members"]],
            ["PASS", "PASS"],
        )
        self.assertEqual(exit_code, 0)

    def test_strict_parallel_case_rejects_compat_member(self):
        with self.assertRaisesRegex(runner.HarnessError, "cannot contain compat"):
            self.load(
                [
                    self.leaf("strict", "pass"),
                    self.leaf("compat", "pass", tier="compat"),
                    {
                        "name": "pressure",
                        "tier": "strict",
                        "members": ["strict", "compat"],
                    },
                ]
            )

    def test_unknown_xfail_name_is_rejected(self):
        cases, _ = self.load([self.leaf("pass", "pass")])
        path = self.root / "xfail.json"
        path.write_text(json.dumps({"passs": "typo"}), encoding="utf-8")
        with self.assertRaisesRegex(runner.HarnessError, "unknown cases: passs"):
            runner.load_xfails(path, cases)

    def test_main_emits_json_lines(self):
        cases, path = self.load([self.leaf("pass", "pass")])
        del cases
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = runner.main(
                ["--bindir", str(self.bindir), "--cases", str(path), "--seed", "9"]
            )
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(exit_code, 0)
        self.assertEqual(records[0]["event"], "case")
        self.assertEqual(records[-1]["event"], "summary")
        self.assertEqual(records[-1]["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
