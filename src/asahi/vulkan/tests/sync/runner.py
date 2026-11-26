#!/usr/bin/env python3
"""Run Borealis synchronization regressions as isolated subprocesses."""

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time


SKIP_RETURN_CODE = 77
# After a timeout kills a test's process group, how long to keep collecting
# its output. A descendant that left the group can hold the pipes open.
KILL_DRAIN_SECONDS = 5.0
VALID_TIERS = {"strict", "compat"}
VALID_STATUSES = {"PASS", "FAIL", "SKIP", "XFAIL", "XPASS", "HARNESS_ERROR"}


class HarnessError(Exception):
    """An invalid suite or an inability to run it."""


def _valid_timeout(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _require_type(value, expected, description):
    if not isinstance(value, expected):
        raise HarnessError("{} has the wrong type".format(description))


def load_cases(path):
    """Load and validate a cases file, returning defaults and cases by name."""
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError("cannot read cases file {}: {}".format(path, error))

    _require_type(document, dict, "cases document")
    if document.get("version") != 1:
        raise HarnessError("cases document version must be 1")

    defaults = document.get("defaults", {})
    _require_type(defaults, dict, "defaults")
    default_timeout = defaults.get("timeout", 60)
    if not _valid_timeout(default_timeout):
        raise HarnessError("defaults.timeout must be positive")

    entries = document.get("cases")
    _require_type(entries, list, "cases")
    cases = {}
    for index, entry in enumerate(entries):
        _require_type(entry, dict, "case {}".format(index))
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise HarnessError("case {} has no valid name".format(index))
        if name in cases:
            raise HarnessError("duplicate case name: {}".format(name))

        tier = entry.get("tier", "strict")
        if tier not in VALID_TIERS:
            raise HarnessError("case {} has invalid tier {}".format(name, tier))
        has_executable = "executable" in entry
        has_members = "members" in entry
        if has_executable == has_members:
            raise HarnessError(
                "case {} must have exactly one of executable or members".format(name)
            )

        normalized = dict(entry)
        normalized["tier"] = tier
        timeout = normalized.get("timeout", default_timeout)
        if not _valid_timeout(timeout):
            raise HarnessError("case {} timeout must be positive".format(name))
        normalized["timeout"] = timeout

        if has_executable:
            executable = normalized["executable"]
            if not isinstance(executable, str) or not executable:
                raise HarnessError("case {} has no valid executable".format(name))
            executable_path = Path(executable)
            if executable_path.is_absolute() or ".." in executable_path.parts:
                raise HarnessError(
                    "case {} executable must be relative to --bindir".format(name)
                )
            args = normalized.get("args", [])
            _require_type(args, list, "case {} args".format(name))
            if not all(isinstance(arg, str) for arg in args):
                raise HarnessError("case {} args must be strings".format(name))
            environment = normalized.get("env", {})
            _require_type(environment, dict, "case {} env".format(name))
            if not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in environment.items()
            ):
                raise HarnessError("case {} env must map strings to strings".format(name))
            if not isinstance(normalized.get("optional", False), bool):
                raise HarnessError("case {} optional must be boolean".format(name))
        else:
            members = normalized["members"]
            _require_type(members, list, "case {} members".format(name))
            if not members or not all(isinstance(member, str) for member in members):
                raise HarnessError("case {} members must be non-empty strings".format(name))

        cases[name] = normalized

    for name, case in cases.items():
        for member in case.get("members", []):
            if member not in cases:
                raise HarnessError("case {} has unknown member {}".format(name, member))
            if "members" in cases[member]:
                raise HarnessError("parallel case {} cannot contain {}".format(name, member))
            if case["tier"] == "strict" and cases[member]["tier"] != "strict":
                raise HarnessError(
                    "strict parallel case {} cannot contain {} case {}".format(
                        name, cases[member]["tier"], member
                    )
                )

    return {"timeout": default_timeout}, cases


def load_xfails(path, case_names):
    """Load an optional mapping from case name to expected-failure reason."""
    if path is None:
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError("cannot read xfail file {}: {}".format(path, error))

    if isinstance(document, list):
        if not all(isinstance(name, str) for name in document):
            raise HarnessError("xfail list entries must be case names")
        xfails = {name: "expected failure" for name in document}
    elif isinstance(document, dict):
        if not all(
            isinstance(name, str) and isinstance(reason, str) and reason
            for name, reason in document.items()
        ):
            raise HarnessError("xfail object must map case names to reasons")
        xfails = dict(document)
    else:
        raise HarnessError("xfail file must contain an object or a list")

    unknown = sorted(set(xfails) - set(case_names))
    if unknown:
        raise HarnessError("xfail file names unknown cases: {}".format(", ".join(unknown)))
    return xfails


def _missing_result(case, iteration, message):
    status = "SKIP" if case.get("optional", False) else "HARNESS_ERROR"
    return {
        "event": "case",
        "name": case["name"],
        "tier": case["tier"],
        "iteration": iteration,
        "status": status,
        "message": message,
    }


def _command_for_case(case, bindir):
    executable = bindir / case["executable"]
    if not executable.is_file() or not os.access(str(executable), os.X_OK):
        return None
    return [str(executable)] + case.get("args", [])


def _run_process(command, environment, timeout):
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(Path(command[0]).parent),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as error:
        return {
            "status": "HARNESS_ERROR",
            "message": "cannot launch {}: {}".format(command[0], error),
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }

    timed_out = False
    pipes_held = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=KILL_DRAIN_SECONDS)
        except subprocess.TimeoutExpired as error:
            # Keep what was read; the test itself is dead.
            pipes_held = True
            stdout = _decode(error.stdout)
            stderr = _decode(error.stderr)
            process.stdout.close()
            process.stderr.close()
            process.wait()

    if timed_out:
        status = "FAIL"
    elif process.returncode == 0:
        status = "PASS"
    elif process.returncode == SKIP_RETURN_CODE:
        status = "SKIP"
    else:
        status = "FAIL"

    result = {
        "status": status,
        "command": command,
        "returncode": process.returncode,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "stdout": stdout,
        "stderr": stderr,
    }
    if timed_out:
        result["timed_out"] = True
        result["message"] = "timed out after {} seconds".format(timeout)
        if pipes_held:
            result["message"] += (
                "; a process outside the test's group kept its output open"
            )
    return result


def _decode(data):
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def run_leaf(case, bindir, timeout, iteration):
    command = _command_for_case(case, bindir)
    if command is None:
        return _missing_result(
            case,
            iteration,
            "executable not found or not executable: {}".format(
                bindir / case["executable"]
            ),
        )

    environment = os.environ.copy()
    environment.update(case.get("env", {}))
    result = _run_process(command, environment, timeout)
    result.update(
        {
            "event": "case",
            "name": case["name"],
            "tier": case["tier"],
            "iteration": iteration,
        }
    )
    return result


def run_parallel(case, cases, bindir, timeout, iteration):
    member_names = case["members"]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(member_names)) as pool:
        futures = [
            pool.submit(run_leaf, cases[member_name], bindir, timeout, iteration)
            for member_name in member_names
        ]
        members = [future.result() for future in futures]

    statuses = {member["status"] for member in members}
    if "HARNESS_ERROR" in statuses:
        status = "HARNESS_ERROR"
    elif "FAIL" in statuses:
        status = "FAIL"
    elif statuses == {"SKIP"}:
        status = "SKIP"
    else:
        status = "PASS"

    return {
        "event": "case",
        "name": case["name"],
        "tier": case["tier"],
        "iteration": iteration,
        "status": status,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "members": members,
    }


def _apply_xfail(result, reason):
    if reason is None:
        return result
    result["xfail_reason"] = reason
    if result["status"] == "PASS":
        result["status"] = "XPASS"
    elif result["status"] == "FAIL":
        result["status"] = "XFAIL"
    return result


def select_cases(cases, requested, include_compat):
    if requested:
        unknown = sorted(set(requested) - set(cases))
        if unknown:
            raise HarnessError("unknown cases: {}".format(", ".join(unknown)))
        return [cases[name] for name in requested]
    return [
        case
        for case in cases.values()
        if case["tier"] == "strict" or include_compat
    ]


def exit_code_for_results(results):
    statuses = [result["status"] for result in results]
    if any(status == "HARNESS_ERROR" for status in statuses):
        return 2
    if any(status in {"FAIL", "XPASS"} for status in statuses):
        return 1
    if statuses and all(status == "SKIP" for status in statuses):
        return SKIP_RETURN_CODE
    if not statuses:
        return SKIP_RETURN_CODE
    return 0


def execute_suite(
    cases,
    bindir,
    requested=None,
    include_compat=False,
    xfails=None,
    repetitions=1,
    timeout_override=None,
    seed=0,
    emit=None,
):
    """Execute selected cases and return (results, process exit code)."""
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise HarnessError("repetitions must be positive")
    if timeout_override is not None and not _valid_timeout(timeout_override):
        raise HarnessError("timeout override must be finite and positive")

    bindir = Path(bindir).resolve()
    if not bindir.is_dir():
        raise HarnessError("--bindir is not a directory: {}".format(bindir))
    selected = select_cases(cases, requested or [], include_compat)
    xfails = xfails or {}
    rng = random.Random(seed)
    results = []
    for iteration in range(1, repetitions + 1):
        ordered = list(selected)
        rng.shuffle(ordered)
        for case in ordered:
            timeout = timeout_override if timeout_override is not None else case["timeout"]
            if "members" in case:
                result = run_parallel(case, cases, bindir, timeout, iteration)
            else:
                result = run_leaf(case, bindir, timeout, iteration)
            result = _apply_xfail(result, xfails.get(case["name"]))
            if result["status"] not in VALID_STATUSES:
                raise HarnessError("internal error: invalid result status")
            results.append(result)
            if emit is not None:
                emit(result)

    return results, exit_code_for_results(results)


def _json_emitter(stream):
    def emit(record):
        json.dump(record, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()

    return emit


def _silence_broken_pipe(stream):
    """Keep interpreter shutdown from retrying a failed stdout flush."""
    try:
        descriptor = stream.fileno()
        null_descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null_descriptor, descriptor)
        finally:
            os.close(null_descriptor)
    except (AttributeError, OSError):
        pass


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindir", required=True, help="directory containing test executables")
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("cases.json")),
        help="suite definition (default: adjacent cases.json)",
    )
    parser.add_argument("--xfail", help="optional JSON expected-failure file")
    parser.add_argument(
        "--include-compat", action="store_true", help="include compatibility cases"
    )
    parser.add_argument(
        "--repetitions", type=int, default=1, help="repeat the selected suite"
    )
    parser.add_argument(
        "--timeout", type=float, help="override every per-case timeout in seconds"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the order cases run in"
    )
    parser.add_argument(
        "--jsonl", default="-", help="JSONL output path, or - for stdout"
    )
    parser.add_argument("--list", action="store_true", help="list case names and exit")
    parser.add_argument("case", nargs="*", help="specific cases to run")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    output = None
    close_output = False
    try:
        if args.jsonl == "-":
            output = sys.stdout
        else:
            output = Path(args.jsonl).open("w", encoding="utf-8")
            close_output = True
        emit = _json_emitter(output)

        _, cases = load_cases(args.cases)
        if args.list:
            for case in cases.values():
                emit(
                    {
                        "event": "case_info",
                        "name": case["name"],
                        "tier": case["tier"],
                        "optional": case.get("optional", False),
                        "description": case.get("description", ""),
                    }
                )
            return 0

        xfails = load_xfails(args.xfail, cases)
        results, result_code = execute_suite(
            cases,
            args.bindir,
            requested=args.case,
            include_compat=args.include_compat,
            xfails=xfails,
            repetitions=args.repetitions,
            timeout_override=args.timeout,
            seed=args.seed,
            emit=emit,
        )
        counts = {status: 0 for status in sorted(VALID_STATUSES)}
        for result in results:
            counts[result["status"]] += 1
        emit(
            {
                "event": "summary",
                "seed": args.seed,
                "repetitions": args.repetitions,
                "counts": counts,
                "exit_code": result_code,
            }
        )
        return result_code
    except BrokenPipeError:
        if output is not None:
            _silence_broken_pipe(output)
        return 0
    except (HarnessError, OSError) as error:
        record = {"event": "harness_error", "message": str(error)}
        if output is None:
            output = sys.stdout
        _json_emitter(output)(record)
        return 2
    finally:
        if close_output and output is not None:
            output.close()


if __name__ == "__main__":
    sys.exit(main())
