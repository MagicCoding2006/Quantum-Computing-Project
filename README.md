# Quant Checklist Scanner - Setup

Finds combinations of market conditions ("checklists") that came before big
stock gains, scores how reliable each one was, re-checks the winners on later
years, and lists which stocks match today. Read
`Quantum_Checklist_Study_Overview.pdf` first for what it does and why.

Setup takes about 5 minutes. You only do steps 1-3 once.


## Mac

1. Unzip and move the `Quant_Checklist_Scanner` folder to your Desktop.

2. Open Terminal (press Cmd + Space, type "Terminal", press Enter).

3. Paste these lines one at a time, pressing Enter after each:

       cd ~/Desktop/Quant_Checklist_Scanner
       python3 -m pip install -r requirements.txt

4. Run it:

       python3 checklist_scanner.py

Next time, open Terminal and run only the `cd` line and step 4.


## Windows

1. Unzip and move the `Quant_Checklist_Scanner` folder to your Desktop.

2. Install Python from https://www.python.org/downloads/ if you don't have it.
   On the first installer screen, CHECK the box "Add python.exe to PATH".

3. Open Command Prompt (press the Windows key, type "cmd", press Enter) and
   paste these lines one at a time:

       cd /d "%USERPROFILE%\Desktop\Quant_Checklist_Scanner"
       py -m pip install -r requirements.txt

4. Run it:

       py checklist_scanner.py

If your Desktop is synced to OneDrive and the `cd` line fails, use:

       cd /d "%USERPROFILE%\OneDrive\Desktop\Quant_Checklist_Scanner"


## If something goes wrong

| Error | Fix |
|---|---|
| `command not found: python3` (Mac) | Run `xcode-select --install`, accept the popup, try again |
| `'py' is not recognized` (Windows) | Reinstall Python and check "Add python.exe to PATH" |
| `No such file or directory` on the `cd` line | Type `cd ` (with a space), drag the folder from Finder/File Explorer into the window, press Enter |
| `externally-managed-environment` on the pip line | Add `--user` to the end of the pip line |

The first run downloads price history, so give it a minute. The full S&P 500
(~500 stocks) takes much longer the first time; after that everything is
saved in `data_cache/` and runs fast.


## Ways to run it

Mac uses `python3`, Windows uses `py`. Everything after that is the same.

    python3 checklist_scanner.py                                   # 10 default stocks
    python3 checklist_scanner.py --universe sp500                  # whole S&P 500
    python3 checklist_scanner.py --sector energy                   # one S&P 500 sector
    python3 checklist_scanner.py --tickers AAPL MSFT NVDA          # your own list
    python3 checklist_scanner.py --vs-market                       # big gain = beat the S&P 500
    python3 checklist_scanner.py --horizon 26 --target 0.25        # +25% over 26 weeks
    python3 checklist_scanner.py --demo                            # fake data, no internet

Sectors: Communication Services, Consumer Discretionary, Consumer Staples,
Energy, Financials, Health Care, Industrials, Information Technology,
Materials, Real Estate, Utilities. Partial names work, e.g. "health" or "tech".


## Reading the results

Everything is saved in the `results_checklist` folder:

- `summary.md` - the full report (opens in any text editor)
- `top_checklists.csv` - the top checklists; opens in Excel or Google Sheets
- `all_checklists.csv` - every checklist the scanner scored
- `partial_matches.csv` - whether meeting only some items of the #1 checklist still worked
- `current_matches.csv` - which stocks meet the top checklists this week

Key columns:
- hit rate - how often a big gain followed when the checklist fired
- base rate - how often a big gain followed on any random week
- lift - hit rate / base rate. 1.0x = no better than random
- confidence floor - the hit rate we're 95% sure it beats, given how often it fired
- test_ columns - the same numbers on later years the checklist was never ranked
  on. These are the ones to trust.


## Adding a checklist item

Add one line to the `CONDITIONS` list near the top of `checklist_scanner.py`.
Each item is one qubit in the Grover version, so ~10 items fits the lab's
11-qubit machine.
