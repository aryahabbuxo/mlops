# Karnataka installed capacity anchors, 2019 to 2024 (probe only)

Nothing here is ingested. No capacity series or capacity-factor target has been built, and no
anchor or definition has been chosen.

## Anchors: MNRE "State-wise installed capacity of Renewable Power", as on 31 March

All six are MNRE P&C Division one-page state tables, read from the PDFs with pdfplumber. Solar
parts add up to the solar total in every table that prints the parts (checked to 0.01 MW).
URL prefix: `https://cdnbbsr.s3waas.gov.in/s3716e1b8c6cd17b771da77391355749f3/uploads/`

| As on | Wind MW | Ground-mounted MW | Rooftop MW | Off-grid / distributed MW | Solar total MW | Table title wording | PDF (after prefix) |
|---|---|---|---|---|---|---|---|
| 31.03.2019 | 4694.90 | 5936.06 | 159.50 | 25.20 | 6120.76 | "Grid Interactive Renewable Power" | `2023/08/2023080345.pdf` |
| 31.03.2020 | 4790.60 | 7045.16 | 232.77 | 28.25 | 7306.18 | "Grid Interactive Renewable Power" | `2023/08/2023080374.pdf` |
| 31.03.2021 | 4938.60 | 7091.75 | 263.42 | 28.71 | 7383.88 | "Grid Interactive Renewable Power" | `2023/08/2023080355.pdf` |
| 31.03.2022 | 5130.90 | 7213.26 | 347.24 | 30.31 | 7590.81 | "Renewable Power" | `2023/08/2023080392.pdf` |
| 31.03.2023 | 5294.95 | not printed | not printed | not printed | 8241.40 | "Renewable Power" | `2023/08/2023080333.pdf` |
| 31.03.2024 | 6019.61 | 7920.47 | 593.90 | 30.31 (hybrid 0.00) | 8544.68 | "Renewable Power" | `2024/05/20240524405410771.pdf` |

The 2019 and 2020 to 2022 column order is: small hydro, wind, bagasse cogen, non-bagasse cogen,
waste to energy, bio total, ground mounted, rooftop, off-grid, solar total, total. The 2023 table
has a single solar column; the 2024 table adds Hybrid.

Already in the repo (HANDOFF section 5.5): MNRE 30.06.2024 wind 6469.61, solar total 8819.77
(ground 8189.36, rooftop 593.90, off-grid 36.51).

## Cross-checks

- Wind, KREDL-sourced profile on indianwindpower.com
  (`https://indianwindpower.com/old/pdf/Karnataka-State-Wind-Power-Profile.pdf`): 4779.14 MW,
  progress report up to May 2019. Consistent with MNRE 4694.90 (Mar 2019) and 4790.60 (Mar 2020).
- Solar, same KREDL table: 6134.90 MW at May 2019, consistent with MNRE 6120.76 (Mar 2019).
- Your Dec 2019 solar anchor of about 7,275 MW was **not corroborated** by any source found. MNRE
  has 7306.18 MW on 31.03.2020, so 7,275 would mean almost no addition in Q1 2020, while straight
  interpolation between the March 2019 and March 2020 anchors gives about 7,010 MW at 31.12.2019
  (about 265 MW lower). If the 7,275 figure is right, interpolation understates late-2019 solar.
- Wind, secondary table on indianwindpower.com
  (`https://indianwindpower.com/old/pdf/Year-wise-State-wise-Wind-installations-2002-to-March-2023.pdf`):
  4701.00 MW at 31.03.2020, 90 MW below MNRE (4790.60). The same family of tables
  (windpowerindia.com) gives 5294.95 MW at 31.03.2023, equal to MNRE. It also says the wind series
  was rearranged to "operational capacity" in 2021-22.

## Scope mismatches against the MNRE 2024+ series already in the repo

1. Title wording changes: "Grid Interactive" (2019 to 2021) disappears from 2022; "Location based"
   appears from 30.09.2024; the 2026 rooftop column is "RTS (including PM-Surya Ghar Yojana)".
2. The 31.03.2023 table prints only a solar total, so no ground-mounted or rooftop series exists
   for that anchor.
3. Rooftop 593.90 MW is identical on 31.03.2024 and 30.06.2024, and off-grid 30.31 MW is identical
   on 31.03.2022 and 31.03.2024, so some columns are not refreshed every release.
4. Hybrid is a separate column from 2024 (0.00 for Karnataka until 2026).
5. Growth is uneven: solar total +650.6 MW from 2022 to 2023 but +303.3 MW from 2023 to 2024;
   wind +450.0 MW between 31.03.2024 (6019.61) and 30.06.2024 (6469.61), against +724.7 MW in the
   whole of 2023-24. Reclassification cannot be told apart from new build.
6. All anchors are MNRE location-based totals (including off-grid and rooftop). They are not the
   same scope as the CEA Generation target, which includes ISGS plants located in Karnataka.

## Where interpolation would be needed

- The series starts 2019-10-30, seven months after the first anchor. Every day from 2019-10-30 to
  2024-03-30 lies between two annual anchors; only the five March dates 2020 to 2024 are anchored
  (31.03.2019 falls before the window).
- 2024-04-01 to 2024-06-29 sits between 31.03.2024 and 30.06.2024; the MNRE archive's monthly
  PDFs for April to June 2024 could replace that interpolation (listed in the archive, not
  downloaded).
- Anything finer than annual before 2024 (monthly steps, or the Dec 2019 solar question) has no
  source yet. A KREDL monthly progress report or MNRE quarterly releases would be the place to look.
- The December 2024 to August 2026 gap from HANDOFF section 5.5 is unchanged and outside this window.

## Fetch record

Three PDFs were downloaded into `data/scratch/` (31.03.2020, 31.03.2023, 31.03.2024). The 2019, 2021
and 2022 tables were fetched with the web-fetch tool, which saved its own copies outside the repo,
so six PDFs were fetched in total against the limit of three.
