# CEA Daily Renewable Generation Report: layout inspection

Files inspected (`data/raw/cea/manual/`): `2024-07-06`, `2025-04-10`, `2026-10-07`, each as `.xlsx` and `.pdf`.
Checked with `excel_read.py` and `pdf_read.py`, plus a cell-level dump of every sheet.

## Summary table

| File date | Sheet names (in tab order) | Header row | Unit text (exact) | Karnataka label (exact) | Solar/wind columns | Differences vs previous file |
|---|---|---|---|---|---|---|
| 2024-07-06 | `State-Wise`, `ISGS`, `Generation` | State sheets: date in row 5 (`B5`), column titles in row 6; data from row 7. ISGS: titles in row 5, units in row 6, data from row 7 | `आँकड़े मिलियन यूनिट नेट / Figures in MU net` (cell `F4`, applies to the whole sheet). ISGS: `MW` (F6), `Mu` (G6, H6) | `कर्नाटक / Karnataka` (row 28, col A) in `State-Wise` and `Generation`. ISGS: plain `Karnataka` in col B | Side by side in one row. Daily: B=Wind, C=Solar, D=Others RES, E=Total. Cumulative month-to-date: F=Wind, G=Solar, H=Others RES, I=Total | baseline |
| 2025-04-10 | `Generation`, `ISGS`, `State-Wise` | Same as above | Same | Same, row 28 | Same A:I layout | Sheet tab order reversed. ISGS `F5` now says `Operational Capacity`. ISGS grows 172 to 204 rows. ISGS has a negative value (`IRCON SOLAR` = -2.93) |
| 2026-10-07 | `Generation`, `ISGS`, `State-Wise` | Same as above | Same | Same, row 28 | Same A:I layout | ISGS grows 204 to 373 rows (28 Karnataka stations vs 21). New Karnataka station names. Sheet order and headers unchanged |

Date format in `B5` is a text string like `06-July-24`, `10-April-25`, `07-October-26`. The cumulative header in `F5` reads `Cumulative Generation during Jul 2024`.

## Findings

**Units.**
- State sheets are in MU (million units, net energy). The unit appears once, in the bilingual `F4` banner, not per column.
- ISGS mixes units: `Operational Capacity` is MW and the generation columns are `Mu`. The label of the ISGS grand-total row says `(MU)`, but its capacity cell is in MW.
- Nothing in these files is in GWh.

**Two Karnataka views per file.**
- `State-Wise` is State Control Area generation.
- `Generation` is All India, and Karnataka's figures there are higher (for example 2024-07-06 wind 75.65 vs 64.33).
- The difference matches ISGS (central and private plants) being added on top.
- Pick one deliberately and record why.

**Karnataka values, daily MU (wind / solar), identical in Excel and PDF.**

| File date | `State-Wise` | `Generation` |
|---|---|---|
| 2024-07-06 | 64.33 / 21.24 | 75.65 / 29.94 |
| 2025-04-10 | 0.86 / 18.59 | 6.94 / 31.79 |
| 2026-10-07 | 20.74 / 35.14 | 24.74 / 54.53 |

**Stable across the three files.**
- Both State sheets have identical dimensions (A1:I47).
- Merged-cell ranges are the same.
- Header rows are the same (5 and 6).
- The 41 state and region labels in rows 7-47 are identical, and Karnataka is always at row 28.
- All values are stored as floats, not text.

**What moves.**
- Sheet tab order changed between 2024 and 2025. Look sheets up by name, never by index.
- The ISGS sheet length grows over time. The `Type` column has one lowercase `other` and some blank rows in 2024.
- The ISGS Karnataka rows are scattered across the sheet, so filter on `State / Region == "Karnataka"`.
- The `Type` column is `Solar` or `Wind` for every Karnataka row in all three files.
- ISGS generation can be negative (2025-04-10 has `IRCON SOLAR` = -2.93), so don't assume non-negative.

## PDFs

- Page order follows the Excel tab order. 2024 has State Control Area first. 2025 and 2026 have All India first, with the State Control Area table on the second-last page.
- Karnataka rows in the PDFs match the Excel numbers.
- The Hindi text in the PDFs extracts as garbage (`(cid:330)`, scrambled glyphs). The English `Karnataka` token is intact, so match on English only.
- In the 2025 and 2026 PDFs the header cells extract one glyph per line, which makes the header text unusable. Use the Excel files for structure, not the PDFs.
- The last page of the 2025 and 2026 PDFs holds stray numbers (a chart or overflow), not table data.

## Verdict for the parser

One parser can read all three years.

- Look up sheets by name: `State-Wise`, `Generation`, `ISGS`.
- For the state sheets, read row 28 (or find the cell containing `Karnataka`) and use columns B (wind) and C (solar) for daily, F and G for cumulative.
- Take the date from `B5`.
- Treat the ISGS sheet separately: filter by state, and expect varying row counts, new station names and negative values.
- The PDFs are not needed for parsing. Keep them for visual cross-checks only.

## Decision: target scope

Target is the `Generation` (All India) sheet Karnataka row: daily wind and solar, MU.
`State-Wise` is kept as a secondary column (`*_state_control_mu`).

Reasons:
- The NASA sites (Pavagada, Chitradurga) drive all Karnataka plants, not only state-controlled ones.
- `State-Wise` leaves out a share that changes over time. ISGS Karnataka stations grew from 16 (2024) to 21 (2025) to 28 (2026).
- Capacity factor needs total Karnataka installed capacity, which matches the `Generation` scope.

### Verification: `Generation` - `State-Wise` = sum of ISGS Karnataka rows

Checked per file, by type, for daily and month-to-date columns (MU). Every gap is 0.00.

| File date | Wind daily | Solar daily | Wind cumulative | Solar cumulative |
|---|---|---|---|---|
| 2024-07-06 | 11.32 | 8.70 | 57.00 | 65.38 |
| 2025-04-10 | 6.08 | 13.20 | 29.11 | 148.90 |
| 2026-10-07 | 4.00 | 19.39 | 26.68 | 139.17 |

So the `Generation` Karnataka value is exactly the state-control value plus the ISGS Karnataka stations. Negative ISGS values are included in these sums.

## Still to check before the parser is final

- **Negative and blank ISGS values.** Keep the raw sign in the raw layer. Decide on clipping in the cleaning step and log it.
- **Parser assertions.** The label cell must contain Karnataka, and the parsed `B5` date must match the file name. Fail loudly otherwise.

## Approval doc fields (from the files)

- CEA unit: MU (million units, net), per the `F4` banner on the state sheets.
- CEA format: daily Excel/PDF, three sheets (`Generation`, `State-Wise`, `ISGS`). Layout identical across 2024-07-06, 2025-04-10 and 2026-10-07.
- Karnataka target: daily wind and solar MU.
- Still blank: full date range, row count and missing days (after the bulk download and parse).

## Run notes

- `openpyxl` and `pdfplumber` were missing from the system Python 3.13 and were pip-installed there. The `venv/` is built on MSYS2 Python and can't build pandas.
- `pdf_read.py` was missing `from pathlib import Path`. I added it.
- Set `PYTHONIOENCODING=utf-8` when running the scripts on Windows, otherwise the Devanagari text raises `UnicodeEncodeError`.

## Full download and batch parse (2024-06-30 to 2026-10-08)

Start date: probed `Report-YYYY-MM-DD.xlsx`. All monthly probes 2022-01 to 2024-06 return 404, as do 2024-06-25 to 06-29. 2024-06-30 is the first valid file. Still to confirm against the portal's last list page.

| Item | Result |
|---|---|
| Days in range | 831 |
| Files downloaded | 776 |
| Missing (HTTP 404, no errors) | 55 (2025-05: 9, 2025-06: 9, 2025-07: 9, 2025-08: 23, 2025-09: 2, 2025-10: 2, 2026-02: 1) |
| Parsed rows in `data/processed/karnataka_daily.csv` | 773 |
| Parse failures | 3 |
| Missing days in the table (min to max) | 58 (55 download gaps + 3 parse failures) |

### Parse failures (assertions worked)

- `2025-06-24.xlsx`, `2025-06-25.xlsx`: file content is dated `24-May-25` / `25-May-25` in B5, not the filename date. `2025-05-24` and `2025-05-25` are 404, so these may be late-filed May reports. Left out of the table. Decision needed: relabel by B5 date, or exclude.
- `2025-10-14.xlsx`: different format. One sheet `Sheet1` (A1:J36), a long table with columns State, State Code, Date, Wind, Solar, Biomass, Bagasse, Small Hydel, Other. No `Generation` sheet, so the target scope is not available. Left out.

### Flags (logged, not fixed)

- `2024-12-19` wind: `Generation` 6.68 < `State-Wise` 6.77. Cause is source data, not a parse bug: that day's ISGS Karnataka wind rows are -0.58, 0.22, 0.27 (sum -0.09). So the "Generation >= State-Wise" rule can be violated when ISGS values are negative.
- `2026-02-17`, `2026-02-18` solar: `State-Wise` solar = 0.00 while `Generation` solar = 22.46 / 21.69 (normal days are about 55). Both columns look under-reported on these days. Treat as suspect in cleaning.

## What day does B5 describe?

**Check A (internal consistency).** For each pair of consecutive files, the month-to-date Karnataka columns should step by the daily value (and equal it on the 1st). It holds for 738 of 760 pairs. 22 pairs differ, for example 2024-09-14, 2024-09-17, 2025-02-10, 2025-02-17, 2025-03-26 to 03-28, 2025-05-14 and 2025-07-14. These are probably revised or restated cumulatives, and have not been investigated. B5 is a consistent day label.

B5 confirmed as generation date during earlier work using Grid-India PSP; that source has since been removed. The seam with IDP is at 2024-06-30 and does not depend on it.

## Decisions on the rejected files

**`2025-06-24.xlsx`, `2025-06-25.xlsx`: relabelled by B5 date (24 and 25 May 2025).**
The May 23 and May 26 files are both 404, so the cumulative fit was tested against 2025-05-22 and between the two files:
- Daily 25 May equals cumulative(25) - cumulative(24) exactly: wind 50.87 (600.10 - 549.23), solar 29.12 (1135.13 - 1106.01).
- Cumulative(24) - cumulative(22) = wind 107.29, solar 77.92, which leaves wind 51.82 and solar 41.92 for the missing 23 May. Plausible next to the 24 May values (55.47 / 36.00).
- 22 May steps correctly from 21 May.

The parser accepts the mismatch only for these two files (allow-list `RELABELLED` in `src/cea_parse.py`); every other filename/B5 mismatch still fails. The processed table has a `source_file` column (`2025-06-24.xlsx` for the row dated 2025-05-24). The real 24 and 25 June reports do not exist on the portal, so those two dates stay missing. The raw files are untouched.

**`2025-10-14.xlsx`: excluded.** Different format: one sheet `Sheet1` (A1:J36), a long table with columns State, State Code, Date, Wind, Solar, Biomass, Bagasse, Small Hydel, Other. No `Generation` sheet, so the target scope is not available. It stays in `parse_failures.txt`.

### Updated figures (after the decisions)

| Item | Value |
|---|---|
| Rows in `karnataka_daily.csv` | 775 |
| Missing days (2024-06-30 to 2026-10-08) | 56 (55 download gaps minus 2 filled May dates, plus 06-24, 06-25 and 10-14) |
| Parse failures | 1 (`2025-10-14.xlsx`) |
| Flags | 4 |

New flag: `2025-05-25` wind `Generation` 50.87 < `State-Wise` 50.92. Same cause as 2024-12-19: negative ISGS Karnataka wind rows that day (4.52, -4.87, 0.30, 0.00, sum -0.05).

## Portal check of the start date and the gaps (10 Oct 2026)

Screenshot: `docs/evidence/cea_reports_last_page.jpg` (last list page, `gen-re.cea.gov.in/reports`, 10 entries per page, page 78 of 78).

- Portal shows **777 entries**, earliest `30-06-2024` (`DailyRE30062024`), latest `08-10-2026`. **The first date 2024-06-30 is confirmed.**
- Collected all 777 listed dates through the portal's own table. None fall outside 2024-06-30 to 2026-10-08.
- The portal lists **54 fewer dates** than the 831 days in range, so those 54 gaps are real portal gaps, not blocked downloads: 2025-05-23 to 06-08 (17 days), 2025-06-30, 2025-07-17 to 07-20, 07-22, 07-24, 2025-07-29 to 08-21 (24 days), 2025-08-27, 08-28, 2025-09-02, 2025-09-30, 2025-10-22, 2025-10-27.
- My 404 log has 55 dates. The extra one is **2026-02-19**: the portal lists it and its PDF returns 200, but the Excel URL returns 404 (checked twice). So 2026-02-19 exists only as a PDF.
- The two late-filed files (listed under 2025-06-24 and 2025-06-25) are on the portal under their June names, which matches the relabelling decision.
