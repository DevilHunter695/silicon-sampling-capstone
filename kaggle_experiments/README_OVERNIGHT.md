# Overnight plan (two Kaggle accounts)

## Account 1: FRIEND (already running `run_seed(0)` interactively, about 3.5 h left)
1. Do nothing to the running cell. Do not stop the session.
2. Add ONE new cell below it, paste the contents of `friend_queue_cell.py`, run it. It queues and starts seed 2 automatically after seed 0
   (about 4 h more). Total about 7.5 h, under the 12 h session limit.
3. Keep the laptop awake with the screen on and the browser tab open (an interactive session is tied to the tab):
       caffeinate -dimsu
   Plug the charger in. Disable any VPN/auto-logout. Do not close the tab.
4. In the morning download from the Output panel: `seed0_results.zip` and `seed2_results.zip` (files in /kaggle/working).
   If the session ended early, whatever zips exist are still useful; also grab `seed2/train_log.jsonl` etc. from the file browser.

## Account 2: YOU (commit mode: runs on Kaggle's servers, your Mac may sleep)
1. kaggle.com -> Datasets -> New dataset (Private) -> upload the CONTENTS of `kaggle_pkg3/` (adapter/, repo/, scripts, parquet, notebooks). Name it `pew-pkg3`.
   If you also have the older `finetune` dataset, do NOT attach it (the notebook looks for the new script by content, but keep it clean).
2. New notebook -> File -> Import -> `overnight_user_account.ipynb`.
3. Settings: Accelerator GPU T4 x2, Internet On. Add Input -> `pew-pkg3`.
4. (Optional 5-minute check) Run cells 1-2 interactively, confirm paths print, then press Save Version -> **Save & Run All (Commit)**.
   Or press Save & Run All directly.
5. Close the tab. Everything runs server-side. Expected 7 to 9 h. Stages: unseen-item transfer (lab adapter, 32 items, 1000 respondents),
   model-scale zero-shot (Qwen2.5-14B, then 32B), breadth seed 1 (A vs B).
6. In the morning open the finished version -> Output: `result_unseen.zip`, `result_scale.zip`, `seed1_results.zip`, `status.json`, `log_*.txt`.

## Safety built in
* Every stage is wrapped in try/except; a failure is recorded in `status.json` and the next stage still runs.
* A zip is written after each stage, so partial results survive if the session is killed at the 12 h limit.
* Training checkpoints every 100 steps and resumes if a process restarts; inference resumes from finished rows.
* Time guards: stages are skipped if they would start too late; every worker has `--max-hours`.
* fp16 NaN guard aborts a worker loudly instead of writing garbage.
* HF cache is under /tmp (not the 20 GB working dir); the 14B/32B caches are deleted after use.
* No secrets are used anywhere.

## If something fails
Send me `status.json` and the tail of the relevant `log_*.txt`. Partial results are still analysable.
