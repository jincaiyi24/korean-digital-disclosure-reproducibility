"""한글 파일명을 검증한 뒤 원래 실행 경로를 복원하는 공개 진입점.

분류·추정 코드는 수정하지 않는다. 기본 동작은 원래 진입점의 오프라인
입력 점검이며, 실제 계산은 사용자가 --run을 지정했을 때만 위임한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
import uuid

MAPPING_SHA256 = "227e1383c43c3751e56f0498f12859e6381e1ec7934c7aecb68ad43d75f47cce"
MAPPING_FILENAME = "파일명대응표.json"
EXPECTED_COUNT = 94
FOLDERS = {"설명서", "코드", "CSV자료"}
ROOT_EXCEPTIONS = {"LICENSE", ".gitignore", ".gitattributes"}
MANIFEST_KEYS = {"schema", "reference_commit", "record_count", "source_scope", "documentation_only_changes_since_reference", "entries"}
ENTRY_KEYS = {"old_relative_path", "new_korean_path", "size", "sha256"}
PATH_FLAGS = ("data_root", "finance_root", "raw_dir", "supplemental_raw_dir", "viewer_capture_root", "output")


class LayoutError(Exception):
    """수치나 비밀값을 포함하지 않는 구조·파일 검증 실패."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_parts(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise LayoutError("안전하지 않은 상대 경로")
    if unicodedata.normalize("NFC", value) != value:
        raise LayoutError("정규화되지 않은 파일명")
    p = PurePosixPath(value)
    parts = value.split("/")
    if p.is_absolute() or any(v in {"", ".", ".."} or v.endswith((".", " ")) for v in parts):
        raise LayoutError("안전하지 않은 경로 요소")
    if any(re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", v) for v in parts):
        raise LayoutError("예약된 파일명")
    return tuple(parts)


def collision_key(value):
    return unicodedata.normalize("NFC", value).casefold()


def ordinary(path, directory=False):
    try:
        info = Path(path).lstat()
    except OSError:
        raise LayoutError("필수 파일 또는 디렉터리를 읽을 수 없음") from None
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise LayoutError("심볼릭 링크 또는 재분석 지점은 허용하지 않음")
    if directory and not stat.S_ISDIR(info.st_mode):
        raise LayoutError("디렉터리 대신 다른 객체가 있음")
    if not directory and not stat.S_ISREG(info.st_mode):
        raise LayoutError("일반 파일이 아님")
    return info


def child(root, relative, must_exist=True):
    parts = safe_parts(relative)
    root = Path(root)
    ordinary(root, True)
    current = root
    for i, part in enumerate(parts):
        current = current / part
        if must_exist:
            ordinary(current, directory=i < len(parts) - 1)
    if not current.resolve().is_relative_to(root.resolve()):
        raise LayoutError("패키지 또는 작업 디렉터리 밖의 경로")
    return current


def validate_mapping(obj):
    if not isinstance(obj, dict) or set(obj) != MANIFEST_KEYS:
        raise LayoutError("알 수 없는 대응표 형식")
    if obj["schema"] != "KOREAN_FLAT_LAYOUT_MAPPING_V1" or obj["record_count"] != EXPECTED_COUNT:
        raise LayoutError("대응표 버전 또는 파일 수 불일치")
    if obj["reference_commit"] != "c040e574ebd4bc3b1b1bcdac3b28831b1034089e":
        raise LayoutError("원래 검증 커밋 불일치")
    rows = obj["entries"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise LayoutError("대응표 파일 수 불일치")
    old_seen, new_seen = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != ENTRY_KEYS:
            raise LayoutError("알 수 없는 파일 레코드")
        old = safe_parts(row["old_relative_path"])
        new = safe_parts(row["new_korean_path"])
        if not (len(new) == 2 and new[0] in FOLDERS) and not (len(new) == 1 and new[0] in ROOT_EXCEPTIONS):
            raise LayoutError("공개 파일은 세 폴더의 바로 아래 또는 표준 루트 파일이어야 함")
        if len(new) == 2 and not re.search(r"[가-힣]", new[1]):
            raise LayoutError("한글 설명이 없는 공개 파일명")
        a, b = collision_key(row["old_relative_path"]), collision_key(row["new_korean_path"])
        if a in old_seen or b in new_seen:
            raise LayoutError("중복되거나 충돌하는 파일 경로")
        old_seen.add(a); new_seen.add(b)
        if type(row["size"]) is not int or row["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise LayoutError("잘못된 파일 크기 또는 해시")
    for paths in (old_seen, new_seen):
        if any(any("/".join(p.split("/")[:i]) in paths for i in range(1, len(p.split("/")))) for p in paths):
            raise LayoutError("파일과 디렉터리 경로가 충돌함")
    return rows


def load_mapping(public_root):
    path = child(public_root, "코드/" + MAPPING_FILENAME)
    if sha(path) != MAPPING_SHA256:
        raise LayoutError("파일명 대응표 해시 불일치")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise LayoutError("대응표를 읽을 수 없음") from None
    return obj, validate_mapping(obj)


def verify_sources(public_root, rows):
    hashes = {}
    for row in rows:
        path = child(public_root, row["new_korean_path"])
        if path.stat().st_size != row["size"] or sha(path) != row["sha256"]:
            raise LayoutError("공개 파일 바이트가 대응표와 다름")
        hashes[row["new_korean_path"]] = row["sha256"]
    return hashes


def restore_tree(public_root, tree, rows):
    ordinary(tree, True)
    if any(Path(tree).iterdir()):
        raise LayoutError("복원 작업 디렉터리가 비어 있지 않음")
    before = verify_sources(public_root, rows)
    for row in rows:
        src = child(public_root, row["new_korean_path"])
        dst = child(tree, row["old_relative_path"], False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Every ancestor belongs to the newly created tree; reject any injected link.
        for parent in [dst.parent, *list(dst.parent.parents)[:len(safe_parts(row["old_relative_path"])) - 1]]:
            ordinary(parent, True)
        with src.open("rb") as source, dst.open("xb") as target:
            shutil.copyfileobj(source, target, 1 << 20)
        ordinary(dst)
        if dst.stat().st_size != row["size"] or sha(dst) != row["sha256"]:
            raise LayoutError("복원 파일 해시 불일치")
    actual = sorted(p.relative_to(tree).as_posix() for p in Path(tree).rglob("*") if p.is_file())
    if actual != sorted(r["old_relative_path"] for r in rows):
        raise LayoutError("복원된 94개 원래 경로 집합 불일치")
    if verify_sources(public_root, rows) != before:
        raise LayoutError("복원 중 공개 파일 변경 감지")
    return before


def parse_args(argv, caller_cwd):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("reproduce", "acquire"), default="reproduce")
    for name in PATH_FLAGS:
        p.add_argument("--" + name.replace("_", "-"), type=Path)
    p.add_argument("--run", action="store_true")
    p.add_argument("--rounds", type=int, choices=(1, 3))
    acquisition = p.add_mutually_exclusive_group()
    acquisition.add_argument("--check-only", action="store_true")
    acquisition.add_argument("--fetch", action="store_true")
    args = p.parse_args(argv)
    if args.mode == "acquire":
        if args.data_root is None:
            p.error("--mode acquire에는 --data-root가 필요합니다")
        if args.output is not None or args.run or args.rounds is not None or any(getattr(args, n) is not None for n in ("finance_root", "raw_dir", "supplemental_raw_dir", "viewer_capture_root")):
            p.error("acquire 모드에는 재현 계산용 인자를 지정할 수 없습니다")
    else:
        if args.output is None:
            p.error("reproduce 모드에는 새 --output이 필요합니다")
        if args.fetch or args.check_only:
            p.error("--fetch와 --check-only는 --mode acquire 전용입니다")
        args.rounds = 1 if args.rounds is None else args.rounds
    for name in PATH_FLAGS:
        value = getattr(args, name)
        if value is not None:
            value = value if value.is_absolute() else Path(caller_cwd) / value
            setattr(args, name, value.resolve())
    return args


def delegated_command(args, tree):
    tree = Path(tree)
    if args.mode == "acquire":
        return [sys.executable, "-B", str(tree / "code/acquire_locked_reports.py"),
                "--bundle", str(tree / "bundle"), "--data-root", str(args.data_root),
                "--fetch" if args.fetch else "--check-only"]
    cmd = [sys.executable, "-B", str(tree / "code/public_run.py"), "--bundle", str(tree / "bundle")]
    for name in PATH_FLAGS:
        value = getattr(args, name)
        if value is not None:
            cmd += ["--" + name.replace("_", "-"), str(value)]
    cmd += ["--rounds", str(args.rounds)]
    if args.run:
        cmd.append("--run")
    return cmd


def owned_cleanup(container, expected_parent, owner_token):
    container, expected_parent = Path(container), Path(expected_parent)
    ordinary(container, True)
    if container.resolve().parent != expected_parent.resolve() or not container.name.startswith("kci_ko_"):
        raise LayoutError("이 실행이 만든 임시 디렉터리가 아님")
    marker = child(container, ".owned-workspace")
    if marker.read_text(encoding="ascii") != owner_token:
        raise LayoutError("임시 작업 디렉터리 소유 표식 불일치")
    for directory, dirs, files in os.walk(container, followlinks=False):
        ordinary(directory, True)
        for name in dirs:
            ordinary(Path(directory) / name, True)
        for name in files:
            ordinary(Path(directory) / name)
    shutil.rmtree(container)


def execute(argv=None, public_root=None, delegate=None):
    caller_cwd = Path.cwd().resolve()
    args = parse_args(argv, caller_cwd)
    root = Path(public_root) if public_root is not None else Path(__file__).resolve().parent.parent
    ordinary(root, True)
    obj, rows = load_mapping(root)
    verify_sources(root, rows)
    parent = Path(tempfile.gettempdir()).resolve()
    ordinary(parent, True)
    container = Path(tempfile.mkdtemp(prefix="kci_ko_", dir=parent))
    token = uuid.uuid4().hex
    marker = container / ".owned-workspace"
    with marker.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(token)
    tree = container / "workspace"
    tree.mkdir()
    try:
        for name in PATH_FLAGS:
            value = getattr(args, name)
            if value is not None and value.is_relative_to(container.resolve()):
                raise LayoutError("사용자 입력·출력은 임시 복원 공간 밖에 있어야 함")
        restore_tree(root, tree, rows)
        command = delegated_command(args, tree)
        run = delegate if delegate is not None else subprocess.run
        completed = run(command, cwd=str(caller_cwd), check=False)
        if verify_sources(root, rows) != {r["new_korean_path"]: r["sha256"] for r in rows}:
            raise LayoutError("위임 실행 중 공개 파일 변경 감지")
        return int(completed.returncode)
    finally:
        owned_cleanup(container, parent, token)


def main():
    try:
        return execute()
    except LayoutError as exc:
        print("한글 배포 파일 검증 실패: " + str(exc), file=sys.stderr)
        return 2
    except Exception:
        # Do not serialize OS errors, network URLs, arguments, or credential-bearing text.
        print("한글 배포 준비 중 오류가 발생했습니다. 기존 원본은 덮어쓰지 않았습니다.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
