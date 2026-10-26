#!/bin/bash
# build docx -> Word PDF -> measure TOC pages -> rebuild -> Word PDF (twice for stability)
set -e
PY=../venv/bin/python
echo '{}' > toc_pages.json
for i in 1 2 3; do
  $PY build_report.py rep_$i.docx toc_pages.json >/dev/null
  cp rep_$i.docx /Users/krishank/Downloads/_rp.docx
  osascript <<OSA >/dev/null 2>&1
with timeout of 190 seconds
tell application "Microsoft Word"
  open POSIX file "/Users/krishank/Downloads/_rp.docx"
  set d to active document
  save as d file name "/Users/krishank/Downloads/_rp.pdf" file format format PDF
  close d saving no
end tell
end timeout
OSA
  mv /Users/krishank/Downloads/_rp.pdf rep_$i.pdf; rm -f /Users/krishank/Downloads/_rp.docx
  cp toc_pages.json toc_prev.json
  python3 tocpages.py rep_$i.pdf | tail -1
  if cmp -s toc_pages.json toc_prev.json && [ $i -gt 1 ]; then echo "stable at pass $i"; cp rep_$i.docx report_final.docx; cp rep_$i.pdf report_final.pdf; break; fi
done
