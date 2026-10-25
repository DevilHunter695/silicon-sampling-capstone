#!/bin/bash
# usage: wordcheck.sh file.docx  -> pages/words and PDF next to scratch dir
SRC="$1"; BASE=$(basename "$SRC" .docx)
cp "$SRC" "/Users/krishank/Downloads/_wc_$BASE.docx"
osascript <<OSA 2>&1 | tail -2
tell application "Microsoft Word"
  open POSIX file "/Users/krishank/Downloads/_wc_$BASE.docx"
  set d to active document
  set n to compute statistics d statistic statistic pages
  save as d file name "/Users/krishank/Downloads/_wc_$BASE.pdf" file format format PDF
  close d saving no
  return (n as string) & " pages"
end tell
OSA
mv "/Users/krishank/Downloads/_wc_$BASE.pdf" "$(dirname "$SRC")/$BASE.pdf"; rm -f "/Users/krishank/Downloads/_wc_$BASE.docx"
