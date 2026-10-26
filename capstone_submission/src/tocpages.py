import subprocess, json, re, sys
pdf = sys.argv[1]; names = json.load(open("toc_names.json"))
n = int(subprocess.run(["gs", "-q", "-dNODISPLAY", "-dNOSAFER", "-c", f"({pdf}) (r) file runpdfbegin pdfpagecount = quit"], capture_output=True, text=True).stdout.strip())
pages = []
for i in range(1, n + 1):
    t = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=txtwrite", f"-dFirstPage={i}", f"-dLastPage={i}", "-sOutputFile=-", pdf], capture_output=True, text=True).stdout
    pages.append(re.sub(r"\s+", " ", t))
contents = next(i for i, t in enumerate(pages) if "CONTENTS" in t[:200])
main0 = next(i for i, t in enumerate(pages) if '1.1 Background and Motivation Public opinion surveys' in t)
res = {}
def norm(s): return re.sub(r"\s+", " ", s).strip()
for nm in names:
    key = norm(nm)[:26]
    for i in range(main0, n):
        if key in pages[i]:
            res[nm] = i; break
    else: res[nm] = None
out = {nm: (str(i - main0 + 1) if i is not None else "?") for nm, i in res.items()}
print({k: v for k, v in out.items()}); json.dump(out, open("toc_pages.json", "w")); print("pages", n, "contents page idx", contents, "main starts pdf page", main0 + 1)
