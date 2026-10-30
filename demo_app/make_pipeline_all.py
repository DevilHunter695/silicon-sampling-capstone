"""Build pipeline_all.html: the pipeline page with all 46,209 test rows embedded (data taken from demo.html)."""
import sys
from pathlib import Path
here = Path(__file__).parent
demo = (here / "demo.html").read_text()
data = demo[demo.index("const D=") + 8: demo.index(";\nconst pc=")]
out = (here / "pipeline_all_template.html").read_text().replace("__DATA__", data, 1)
(here / (sys.argv[1] if len(sys.argv) > 1 else "pipeline_all.html")).write_text(out)
print("wrote", len(out) // 1024, "KB")
