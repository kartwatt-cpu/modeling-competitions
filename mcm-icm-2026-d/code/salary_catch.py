import re
import time
import requests
import pandas as pd

BASE_URL = "https://herhoopstats.com/salary-cap-sheet/wnba/players/salary_{year}/stats_{stats_year}/"

def fetch_wnba_salaries_by_player(year: int, stats_year: int | None = None,
                                 out_csv: str | None = None,
                                 sleep_sec: float = 1.0) -> pd.DataFrame:
    """
    Scrape Her Hoop Stats 'WNBA Salaries by Player' table for a given salary year.
    - year: salary season year (e.g., 2025)
    - stats_year: the 'stats_YYYY' part in URL; often use previous season (e.g., 2024 for salary_2025)
      If None -> defaults to year-1.
    """
    if stats_year is None:
        stats_year = year - 1

    url = BASE_URL.format(year=year, stats_year=stats_year)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; WNBA-Salary-Scraper/1.0; +https://example.com/bot)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    time.sleep(sleep_sec)  # polite rate limiting
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    # read_html usually抓得到主表；如果页面里有多个表，取“列最多”的那个作为主表
    tables = pd.read_html(r.text)
    if not tables:
        raise RuntimeError("No HTML tables found. Page structure may have changed.")

    df = max(tables, key=lambda t: t.shape[1]).copy()

    # --- 清洗：去掉奇怪的音频提示、空格、货币符号 ---
    def clean_cell(x):
        if pd.isna(x):
            return x
        s = str(x)
        s = s.replace("This HTML5 audio player is not supported by your browser", "")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    df = df.applymap(clean_cell)

    # 把所有像 "$202,000" 的列转成数值（如果你需要）
    money_cols = [c for c in df.columns if re.search(r"\b20\d{2}\b", str(c)) or "Salary" in str(c)]
    for c in money_cols:
        df[c] = (
            df[c].astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # 加一个抓取年份标签，方便后续多年的表拼接
    df.insert(0, "salary_year", year)

    if out_csv:
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return df


if __name__ == "__main__":
    # 示例：抓 2025 薪资（页面 URL 里常对应 stats_2024）
    df_2025 = fetch_wnba_salaries_by_player(2025, stats_year=2024, out_csv="wnba_salaries_2025.csv")
    print(df_2025.head())
    print("rows:", len(df_2025), "cols:", len(df_2025.columns))
