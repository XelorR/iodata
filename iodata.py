# coding: utf-8

import os
import re
import csv
from pathlib import Path
import sqlite3
import xlrd
import numpy as np
import pandas as pd


def locate_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    in case if all column names are 'Unnamed'
    it will try to find first row with values
    and set it as header

    then it will try to recognize numbers
    """
    if len([c for c in df.columns if "Unnamed" in c]) >= 2:
        first_row = (df.count(axis=1) >= df.shape[1]).idxmax()
        df.columns = df.loc[first_row]
        df = df.loc[first_row + 1 :]
        for i in df.columns:
            try:
                df[i] = df[i].astype("float64")
            except:
                pass
    return df


def load_data(filepath_in: str) -> pd.DataFrame:
    filetypes = {
        "h5": load_hdf,
        "hdf": load_hdf,
        "hdf5": load_hdf,
        "zstd": load_hdf,
        "db": load_sqlite,
        "sql": load_sqlite,
        "sqlite": load_sqlite,
        "pkl": load_pickle,
        "pkl.zip": load_pickle,
        "pickle": load_pickle,
        "pickle.zip": load_pickle,
        "parquet": load_parquet,
        "feather": load_feather,
        "csv": load_csv,
        "csv.zip": load_csv,
        "tsv": load_csv,
        "tsv.zip": load_csv,
        "xlsx": load_excel,
        "xls": load_excel,
    }
    for key in filetypes.keys():
        if filepath_in.endswith(key):
            return filetypes[key](filepath_in)


def save_data(df: pd.DataFrame, filepath_out: str):
    filetypes = {
        "h5": save_hdf,
        "hdf": save_hdf,
        "hdf5": save_hdf,
        "zstd": save_hdf,
        "db": save_sqlite,
        "sql": save_sqlite,
        "sqlite": save_sqlite,
        "pkl": save_pickle,
        "pkl.zip": save_pickle,
        "pickle": save_pickle,
        "pickle.zip": save_pickle,
        "parquet": save_parquet,
        "feather": save_feather,
        "csv": save_csv,
        "csv.zip": save_csv,
        "tsv": save_csv,
        "tsv.zip": save_csv,
        "xlsx": save_excel,
        "xls": save_excel,
    }
    for key in filetypes.keys():
        if filepath_out.endswith(key):
            return filetypes[key](df, filepath_out)


def load_hdf(filepath_in: str) -> pd.DataFrame:
    """
    extensions: h5, hdf, hdf5, hdf5.zstd
    """
    return pd.read_hdf(filepath_in)


def load_sqlite(filepath_in: str) -> pd.DataFrame:
    """
    extensions: db, sqlite
    """
    cnx = sqlite3.connect(filepath_in)
    df = pd.read_sql_query("SELECT * FROM db", con=cnx)
    cnx.close()
    return df


def load_csv(filepath_in: str) -> pd.DataFrame:
    """
    extensions: csv, csv.zip, tsv, tsv.zip

    will autolocate header
    """
    if filepath_in.endswith("csv") or filepath_in.endswith("csv.zip"):
        return pd.read_csv(filepath_in).pipe(locate_header)
    else:
        return pd.read_csv(filepath_in, sep="\t").pipe(locate_header)


def load_parquet(filepath_in: str) -> pd.DataFrame:
    """
    extension: parquet
    """
    return pd.read_parquet(filepath_in)


def load_feather(filepath_in: str) -> pd.DataFrame:
    """
    extension: feather
    """
    return pd.read_feather(filepath_in)


def load_excel(filepath_in: str) -> pd.DataFrame:
    """
    extensions: xlsx, xls

    will autolocate header

    in case if file exceeds 100MB
    than file will be converted to csv first
    and then csv will be readed to dataframe
    """
    size = os.path.getsize(filepath_in)
    if size >= 1e8:
        return load_excel_large(filepath_in).reset_index().drop("index", axis=1)
    else:
        return pd.read_excel(filepath_in).pipe(locate_header).reset_index().drop("index", axis=1)


def load_excel_large(filepath_in: str) -> pd.DataFrame:
    """
    will be used by load_excel if filesize exceeds 100MB
    """

    def __convert_excel2csv(filepath_in: str):
        filepath_out = "temp.csv"
        p = Path(filepath_out)
        if p.exists():
            os.remove(p)
        wb = xlrd.open_workbook(filepath_in)
        sh = wb.sheet_by_index(0)
        output_csv_file = open(filepath_out, "w", encoding="utf-8")
        wr = csv.writer(output_csv_file, quoting=csv.QUOTE_MINIMAL)
        for rownum in range(sh.nrows):
            wr.writerow(sh.row_values(rownum))
        output_csv_file.close()

    __convert_excel2csv(filepath_in)

    df = load_csv("temp.csv")
    os.remove("temp.csv")
    return df


def load_pickle(filepath_in: str) -> pd.DataFrame:
    if filepath_in.endswith("pickle.zip"):
        return pd.read_pickle(filepath_in, compression="zip")
    else:
        return pd.read_pickle(filepath_in)


def save_hdf(df: pd.DataFrame, filepath_out: str):
    if filepath_out.endswith("zstd"):
        df.to_hdf(filepath_out, "df", format="table", complib="blosc:zstd", complevel=9)
    else:
        df.to_hdf(filepath_out, "df", format="table", complib="blosc:lz4", complevel=9)


def save_sqlite(df: pd.DataFrame, filepath_out: str):
    p = Path(filepath_out)
    if p.exists():
        os.remove(p)
    cnx = sqlite3.connect(filepath_out)
    df.to_sql("db", con=cnx, index=False)
    cnx.close()


def save_csv(df: pd.DataFrame, filepath_out: str):
    if filepath_out.endswith("csv.zip"):
        df.to_csv(filepath_out, encoding="utf-8", compression="zip", index=False)
    elif filepath_out.endswith("tsv"):
        df.to_csv(filepath_out, encoding="utf-8", sep="\t", index=False)
    elif filepath_out.endswith("tsv.zip"):
        df.to_csv(
            filepath_out, encoding="utf-8", sep="\t", compression="zip", index=False
        )
    else:
        df.to_csv(filepath_out, encoding="utf-8", index=False)


def save_parquet(df: pd.DataFrame, filepath_out: str):
    df.to_parquet(filepath_out)


def save_feather(df: pd.DataFrame, filepath_out: str):
    df.reset_index().drop("index", axis=1).to_feather(filepath_out)


def save_excel(df: pd.DataFrame, filepath_out: str):
    if (df.shape[0] >= 1_000_000) and (filepath_out.endswith("xlsx")):
        i = 0
        for d in np.array_split(df, int(df.shape[0] / 750_000) + 1):
            i += 1
            d.to_excel(f"{filepath_out[:-5]}_part{i}{filepath_out[-5:]}", index=False)
    elif (df.shape[0] >= 65000) and (filepath_out.endswith("xls")):
        i = 0
        for d in np.array_split(df, int(df.shape[0] / 60000) + 1):
            i += 1
            d.to_excel(f"{filepath_out[:-4]}_part{i}{filepath_out[-4:]}", index=False)
    else:
        df.to_excel(filepath_out, index=False)


def save_pickle(df: pd.DataFrame, filepath_out: str):
    if filepath_out.endswith("zip"):
        df.to_pickle(filepath_out, compression="zip")
    else:
        df.to_pickle(filepath_out)
