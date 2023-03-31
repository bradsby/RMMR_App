import csv
import datetime as dt
import os

import delta_E
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sample Identify",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def convert_df(df: pd.DataFrame):
    return df.to_csv()


def uniquify(path: str) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1
    path = f"{filename}.{str(counter)}{extension}"
    while os.path.exists(path):
        counter += 1
        path = f"{filename}.{str(counter)}{extension}"
    return path


def find_headers(df):
    if ("L_AVERAGE" in df.columns) and ("L_AVERAGE_WET" in df.columns):
        if df["L_AVERAGE"].isna().sum() <= df["L_AVERAGE_WET"].isna().sum():
            color_labels = ["L_AVERAGE", "A_AVERAGE", "B_AVERAGE"]
            psd_labels = ["PSD"]
        else:
            color_labels = ["L_AVERAGE_WET", "A_AVERAGE_WET", "B_AVERAGE_WET"]
            psd_labels = [
                "45_MICRON_CAMBRIA_MICROTRAC",
                "30_MICRON_CAMBRIA_MICROTRAC",
                "10_MICRON_CAMBRIA_MICROTRAC",
                "2_MICRON_CAMBRIA_MICROTRAC",
                "D10_CAMBRIA_MICROTRAC",
                "D50_CAMBRIA_MICROTRAC",
                "D90_CAMBRIA_MICROTRAC",
            ]
    elif "L_AVERAGE" in df.columns:
        color_labels = ["L_AVERAGE", "A_AVERAGE", "B_AVERAGE"]
        psd_labels = ["PSD"]
    elif "L_AVERAGE_WET" in df.columns:
        color_labels = ["L_AVERAGE_WET", "A_AVERAGE_WET", "B_AVERAGE_WET"]
        psd_labels = [
            "45_MICRON_CAMBRIA_MICROTRAC",
            "30_MICRON_CAMBRIA_MICROTRAC",
            "10_MICRON_CAMBRIA_MICROTRAC",
            "2_MICRON_CAMBRIA_MICROTRAC",
            "D10_CAMBRIA_MICROTRAC",
            "D50_CAMBRIA_MICROTRAC",
            "D90_CAMBRIA_MICROTRAC",
        ]

    base_headers = ["LOT", "BAG_NUMBERS", "PO_OR_BOL"]

    color_headers = base_headers.copy()
    color_headers.extend(color_labels)

    psd_headers = base_headers.copy()
    psd_headers.extend(psd_labels)

    return color_labels, psd_labels, color_headers, psd_headers


def find_parameters(material_type, test):
    if material_type == "Grit" and test == "Color":
        parameters = ["L_AVERAGE", "A_AVERAGE", "B_AVERAGE"]
    elif material_type == "Grit" and test == "PSD":
        parameters = ["PSD"]
    elif material_type == "Powder" and test == "Color":
        parameters = [
            "L_AVERAGE_WET",
            "A_AVERAGE_WET",
            "B_AVERAGE_WET",
        ]
    elif material_type == "Powder" and test == "PSD":
        parameters = [
            "45_MICRON_CAMBRIA_MICROTRAC",
            "30_MICRON_CAMBRIA_MICROTRAC",
            "10_MICRON_CAMBRIA_MICROTRAC",
            "2_MICRON_CAMBRIA_MICROTRAC",
        ]

    return parameters


def format_headers(df):
    df.columns = df.columns.str.replace(" ", "_").str.upper()
    return df


def clean_proficient_data(df, headers):
    return df[headers].dropna().drop_duplicates(["LOT", "BAG_NUMBERS"])


def calc_lot_averages(df_rm, df_prof, headers, labels):
    df_lot_avgs = df_prof.groupby("LOT")[labels].mean()

    df_fill_blanks = df_rm.merge(
        right=df_lot_avgs,
        left_on="LOT",
        right_on="LOT",
        how="left",
        suffixes=("", "_x"),
    )

    return df_fill_blanks


def merge_tests_and_averages(df_rm, df_prof, df_blanks, labels, test):
    df_merged = df_rm.merge(
        right=df_prof,
        left_on=["LOT", "BAG"],
        right_on=["LOT", "BAG_NUMBERS"],
        how="left",
    )
    df_merged[f"{test.upper()}_RESULT_SOURCE"] = "Calculated"
    df_merged.loc[
        df_merged[labels[0]].notna(), f"{test.upper()}_RESULT_SOURCE"
    ] = "Tested"
    df_merged.update(df_blanks, overwrite=False)

    return df_merged


def set_inventory_dtypes(df):
    df = df.fillna(0)
    df["PHYSICAL_FORMAT"] = df["PHYSICAL_FORMAT"].astype("category")
    df["ITEM_DESCRIPTION"] = df["ITEM_DESCRIPTION"].astype("category")
    df["ITEM"] = df["ITEM"].astype("int64")
    df["LOT"] = df["LOT"].astype("string")
    df["BAG"] = df["BAG"].astype("int16")
    df["LOT_NUMBER"] = df["LOT_NUMBER"].astype("string")
    df["QTY_KG"] = df["QTY_KG"].astype("float64")
    df["QTY_LB"] = df["QTY_LB"].astype("float64")
    df["LOCATION"] = df["LOCATION"].astype("category")
    df["LOCATOR"] = df["LOCATOR"].astype("string")
    df["DATE_RECEIVED"] = pd.to_datetime(df["DATE_RECEIVED"])
    df["LAST_CHANGE_DATE"] = pd.to_datetime(df["LAST_CHANGE_DATE"])
    df["QA_STATUS"] = df["QA_STATUS"].astype("category")
    df["LOG_MESSAGE"] = df["LOG_MESSAGE"].astype("string")
    return df


def set_proficient_dtypes(df):
    df = df.fillna(0)
    df["PART"] = df["PART"].astype("category")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["LOT"] = df["LOT"].astype("string")
    df["BAG_NUMBERS"] = df["BAG_NUMBERS"].astype("int16")
    return df


def run_tab1(df_rm, df_prof):
    color_labels, psd_labels, color_headers, psd_headers = find_headers(df_prof)

    df_prof["BAG_NUMBERS"] = pd.to_numeric(df_prof["BAG_NUMBERS"], errors="coerce")

    df_color = clean_proficient_data(df_prof, color_headers)
    df_psd = clean_proficient_data(df_prof, psd_headers)

    df_prof_color_fill_blanks = calc_lot_averages(
        df_rm, df_color, color_headers, color_labels
    )
    df_prof_psd_fill_blanks = calc_lot_averages(df_rm, df_psd, psd_headers, psd_labels)

    df_merged = merge_tests_and_averages(
        df_rm, df_color, df_prof_color_fill_blanks, color_labels, "Color"
    )
    df_merged = merge_tests_and_averages(
        df_merged, df_psd, df_prof_psd_fill_blanks, psd_labels, "PSD"
    )

    df_merged = df_merged.drop(
        ["LOT_NUMBER", "PO_OR_BOL_x", "BAG_NUMBERS_x", "PO_OR_BOL_y", "BAG_NUMBERS_y"],
        axis=1,
    )

    now = dt.datetime.now().strftime("%y%m%d")
    output = f"PROFxINV.{now}.csv"
    output = uniquify(output)

    st.dataframe(df_merged)

    st.download_button(
        label="Download data as CSV",
        data=convert_df(df_merged),
        file_name=output,
        mime="text/csv",
    )

    return df_merged


def run_tab2(df):
    material_type = st.selectbox("Material Type", ["Grit", "Powder"])
    test = "Color"  # st.selectbox("QA Test", ["Color", "PSD"])
    number_of_options = st.number_input(label="Number of Options", min_value=1, value=5)

    parameters = find_parameters(material_type, test)

    n = len(parameters)
    spec_table = pd.DataFrame(
        {
            "Parameter": parameters,
            "Low_Spec": [None] * n,
            "High_Spec": [None] * n,
            "Find_Low": [False] * n,
            "Find_High": [False] * n,
        }
    ).set_index("Parameter")

    edited_table = st.experimental_data_editor(spec_table)

    edited_table["Low_Spec"] = pd.to_numeric(edited_table["Low_Spec"])
    edited_table["High_Spec"] = pd.to_numeric(edited_table["High_Spec"])
    edited_table["Average_Spec"] = edited_table[["Low_Spec", "High_Spec"]].mean(axis=1)

    try:
        specs = edited_table.to_dict()
        df_samples = pd.DataFrame()
        center = edited_table["Average_Spec"].to_list()
        for idx, parameter in enumerate(parameters):
            for bound in ["Low", "High"]:
                if specs[f"Find_{bound}"][parameter]:
                    df_temp = df[
                        (df[parameters[0]].notna())
                        & (df["COLOR_RESULT_SOURCE"] == "Tested")
                    ].copy()
                    target = center.copy()
                    target[idx] = specs[f"{bound}_Spec"][parameter]
                    df_temp["dE"] = df_temp.apply(
                        lambda x: delta_E.dE00(
                            x[parameters[0]],
                            x[parameters[1]],
                            x[parameters[2]],
                            target[0],
                            target[1],
                            target[2],
                        ),
                        axis=1,
                    )
                    df_temp["TARGET_L"] = target[0]
                    df_temp["TARGET_A"] = target[1]
                    df_temp["TARGET_B"] = target[2]

                    df_temp = df_temp.sort_values("dE", ascending=True).head(
                        number_of_options
                    )
                    df_temp["REASON"] = f"{bound.upper()}_{parameter}"
                    df_samples = pd.concat([df_samples, df_temp])

        df_samples = df_samples[
            [
                "ITEM_DESCRIPTION",
                "ITEM",
                "LOT",
                "BAG",
                "LOCATION",
                "QA_STATUS",
                "LOG_MESSAGE",
                parameters[0],
                parameters[1],
                parameters[2],
                "TARGET_L",
                "TARGET_A",
                "TARGET_B",
                "dE",
                "REASON",
            ]
        ]
        st.dataframe(df_samples, use_container_width=True)
        
        now = dt.datetime.now().strftime("%y%m%d%s")
        output = f"spec_dev_samples.{now}.csv"

        st.download_button(
            label="Download data as CSV",
            data=convert_df(df_samples),
            file_name=output,
            mime="text/csv",
        )

    except KeyError:
        st.warning("Please select target specification(s).")


def main():
    path1 = st.sidebar.file_uploader(
        "Upload RM Inventory data", type="csv", accept_multiple_files=False
    )

    path2 = st.sidebar.file_uploader(
        "Upload Proficient data", type="txt", accept_multiple_files=False
    )

    tab_names = ["Proficient & Inventory Merger", "Ideal Samples"]

    tab1, tab2 = st.tabs(tab_names)

    with tab1:
        if path1 and path2:
            df_rm = format_headers(pd.read_csv(path1, thousands=","))

            df_prof = format_headers(
                pd.read_csv(
                    path2, sep="\t", parse_dates=["Date"], quoting=csv.QUOTE_NONE
                )
            )

            df_merged = run_tab1(df_rm, df_prof)

        elif not path1 and not path2:
            st.warning("Upload RM Inventory and Proficient data.")
        elif not path1:
            st.warning("Upload RM Inventory data.")
        elif not path2:
            st.warning("Upload Proficient data.")
    with tab2:
        if path1 and path2:
            run_tab2(df_merged)

        elif not path1 and not path2:
            st.warning("Upload RM Inventory and Proficient data.")
        elif not path1:
            st.warning("Upload RM Inventory data.")
        elif not path2:
            st.warning("Upload Proficient data.")


if __name__ == "__main__":
    main()
