from multiprocessing import freeze_support
from ningsi_capture.__main__ import main

if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
