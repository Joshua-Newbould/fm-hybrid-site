#!/usr/bin/env python3
"""Flag AI-writing signs in .md, .txt and .html files.

Usage: tools/ai_lint.py [--signs PATH] [--list] <files...>

Exit codes: 0 no hits, 1 at least one hit, 2 usage error or unreadable file.
"""
import argparse
import json
import os
import re
import sys
from html.parser import HTMLParser

DEFAULT_SIGNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_signs.json")


class _UsageError(Exception):
    pass


class _ArgParser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(message)


class _TextNodes(HTMLParser):
    """Collect (line, text) for every text node outside <script> and <style>."""

    SKIP = {"script", "style"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.nodes.append((self.getpos()[0], data))


def load_signs(path):
    with open(path, encoding="utf-8") as fh:
        entries = json.load(fh)
    signs = []
    for e in entries:
        signs.append((e["pattern"], e["category"], re.compile(e["pattern"], re.IGNORECASE)))
    return signs


def text_lines(path, content):
    """Yield (line_number, line_text) pairs to scan."""
    if path.lower().endswith((".html", ".htm")):
        parser = _TextNodes()
        parser.feed(content)
        parser.close()
        for start, data in parser.nodes:
            for offset, line in enumerate(data.splitlines(keepends=True)):
                yield start + offset, line
    else:
        for n, line in enumerate(content.splitlines(keepends=True), 1):
            yield n, line


def scan(path, content, signs):
    hits = []
    for lineno, line in text_lines(path, content):
        for pattern, category, rx in signs:
            for m in rx.finditer(line):
                hits.append((lineno, pattern, category, m.group(0).strip()))
    hits.sort(key=lambda h: h[0])
    return hits


def main(argv=None):
    parser = _ArgParser(prog="ai_lint.py", add_help=True)
    parser.add_argument("--signs", default=DEFAULT_SIGNS)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("files", nargs="*")
    try:
        args = parser.parse_args(argv)
        if not args.list and not args.files:
            raise _UsageError("no input files")
    except _UsageError as exc:
        print("usage: tools/ai_lint.py [--signs PATH] [--list] <files...>", file=sys.stderr)
        print("error: %s" % exc, file=sys.stderr)
        return 2

    try:
        signs = load_signs(args.signs)
    except (OSError, ValueError, KeyError, TypeError, re.error) as exc:
        print("error: cannot load signs %s: %s" % (args.signs, exc), file=sys.stderr)
        return 2

    if args.list:
        for pattern, category, _ in signs:
            print("%s\t%s" % (category, pattern))
        return 0

    contents = []
    for path in args.files:
        try:
            with open(path, encoding="utf-8") as fh:
                contents.append((path, fh.read()))
        except (OSError, UnicodeDecodeError) as exc:
            print("error: cannot read %s: %s" % (path, exc), file=sys.stderr)
            return 2

    any_hit = False
    for path, content in sorted(contents, key=lambda pc: pc[0]):
        for lineno, pattern, category, text in scan(path, content, signs):
            any_hit = True
            print('%s:%d: %s  -- %s: "%s"' % (path, lineno, pattern, category, text))
    return 1 if any_hit else 0


if __name__ == "__main__":
    sys.exit(main())
