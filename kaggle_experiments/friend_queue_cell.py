# Paste into a NEW cell in the friend's already-running notebook and run it NOW.
# It queues behind run_seed(0) and starts automatically when seed 0 is finished (no input needed).
try:
    run_seed(2)
except Exception as e:
    import traceback; traceback.print_exc()
    print("run_seed(2) failed:", e)
