import pandas as pd, io

def load(path):
    body = open(path).read().split("-END HEADER-")[1]
    df = pd.read_csv(io.StringIO(body)).replace(-999, pd.NA)
    df["date"] = pd.to_datetime(
        df.rename(columns={"YEAR": "year", "MO": "month", "DY": "day"})[["year", "month", "day"]]
    )
    return df

for p in ["data/raw/nasa/nasa_solar_pavagada.csv", "data/raw/nasa/nasa_wind_chitradurga.csv"]:
    d = load(p)
    print(p, len(d), d["date"].min().date(), d["date"].max().date())
    print(d.isna().sum().to_dict())