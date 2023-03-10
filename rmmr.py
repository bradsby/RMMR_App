import datetime as dt
import os
from io import BytesIO

from kneed import KneeLocator
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import distance
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st
import csv


@st.cache
def convert_df(df: pd.DataFrame):
    """
    Convert df to csv and encode to be able to download.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Encoded dataframe for downloading.

    """
    return df.to_csv()


def uniquify(path: str) -> str:
    """
    Check path for existing file. If file exists, increase counter in file
    path by 1 until file does not already exist.

    Parameters
    ----------
    path : str

    Returns
    -------
    path : str
        Updated with new counter.

    """
    filename, extension = os.path.splitext(path)
    counter = 1
    path = f"{filename}.{str(counter)}{extension}"
    while os.path.exists(path):
        counter += 1
        path = f"{filename}.{str(counter)}{extension}"
    return path


def find_headers(df):
    if ("L_average" in df.columns) and ("L_Average_Wet" in df.columns):
        if df["L_average"].isna().sum() <= df["L_Average_Wet"].isna().sum():
            color_labels = ["L_average", "A_average", "B_Average"]
            psd_labels = ["PSD"]
        else:
            color_labels = ["L_Average_Wet", "A_Average_Wet", "B_Average_Wet"]
            psd_labels = [
                "45_Micron_Cambria_Microtrac",
                "30_Micron_Cambria_Microtrac",
                "10_Micron_Cambria_Microtrac",
                "2_Micron_Cambria_Microtrac",
                "D10_Cambria_Microtrac",
                "D50_Cambria_Microtrac",
                "D90_Cambria_Microtrac",
            ]
    elif "L_average" in df.columns:
        color_labels = ["L_average", "A_average", "B_Average"]
        psd_labels = ["PSD"]
    elif "L_Average_Wet" in df.columns:
        color_labels = ["L_Average_Wet", "A_Average_Wet", "B_Average_Wet"]
        psd_labels = [
            "45_Micron_Cambria_Microtrac",
            "30_Micron_Cambria_Microtrac",
            "10_Micron_Cambria_Microtrac",
            "2_Micron_Cambria_Microtrac",
            "D10_Cambria_Microtrac",
            "D50_Cambria_Microtrac",
            "D90_Cambria_Microtrac",
        ]

    base_headers = ["Lot", "Bag_Numbers", "PO_or_BOL"]

    color_headers = base_headers.copy()
    color_headers.extend(color_labels)

    psd_headers = base_headers.copy()
    psd_headers.extend(psd_labels)

    return color_labels, psd_labels, color_headers, psd_headers


def find_parameters(material_type, test):
    if material_type == "Grit" and test == "Color":
        parameters = ["L_average", "A_average", "B_Average"]
    elif material_type == "Grit" and test == "PSD":
        parameters = ["PSD"]
    elif material_type == "Powder" and test == "Color":
        parameters = [
            "L_Average_Wet",
            "A_Average_Wet",
            "B_Average_Wet",
        ]
    elif material_type == "Powder" and test == "PSD":
        parameters = [
            "45_Micron_Cambria_Microtrac",
            "30_Micron_Cambria_Microtrac",
            "10_Micron_Cambria_Microtrac",
            "2_Micron_Cambria_Microtrac",
        ]

    return parameters


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        space = spacing
        va = "bottom"

        if y_value < 0:
            space *= -1
            va = "top"

        label = "{:.1f}".format(y_value)

        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha="center",
            va=va,
        )


def format_headers(df):
    df.columns = df.columns.str.replace(" ", "_")
    return df


def clean_proficient_data(df, headers):
    return df[headers].dropna().drop_duplicates(["Lot", "Bag_Numbers"])


def calc_lot_averages(df_rm, df_prof, headers, labels):
    df_lot_avgs = df_prof.groupby("Lot")[labels].mean()

    df_fill_blanks = df_rm.merge(
        right=df_lot_avgs,
        left_on="Lot",
        right_on="Lot",
        how="left",
        suffixes=("", "_x"),
    )

    return df_fill_blanks


def merge_tests_and_averages(df_rm, df_prof, df_blanks, labels, test):
    df_merged = df_rm.merge(
        right=df_prof,
        left_on=["Lot", "Bag"],
        right_on=["Lot", "Bag_Numbers"],
        how="left",
    )
    df_merged[f"{test}_Result_Source"] = "Calculated"
    df_merged.loc[df_merged[labels[0]].notna(), f"{test}_Result_Source"] = "Tested"
    df_merged.update(df_blanks, overwrite=False)

    return df_merged


def set_inventory_dtypes(df):
    df = df.fillna(0)
    df["Physical_Format"] = df["Physical_Format"].astype("category")
    df["Item_Description"] = df["Item_Description"].astype("category")
    df["Item"] = df["Item"].astype("int64")
    df["Lot"] = df["Lot"].astype("string")
    df["Bag"] = df["Bag"].astype("int16")
    df["LOT_NUMBER"] = df["LOT_NUMBER"].astype("string")
    df["Qty_kg"] = df["Qty_kg"].astype("float64")
    df["Qty_lb"] = df["Qty_lb"].astype("float64")
    df["Location"] = df["Location"].astype("category")
    df["Locator"] = df["Locator"].astype("string")
    df["Date_Received"] = pd.to_datetime(df["Date_Received"])
    df["Last_Change_Date"] = pd.to_datetime(df["Last_Change_Date"])
    df["QA_Status"] = df["QA_Status"].astype("category")
    df["Log_Message"] = df["Log_Message"].astype("string")
    return df


def set_proficient_dtypes(df):
    df = df.fillna(0)
    df["Part"] = df["Part"].astype("category")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Lot"] = df["Lot"].astype("string")
    df["Bag_Numbers"] = df["Bag_Numbers"].astype("int16")
    return df


def run_tab1(df_rm):
    material = st.selectbox("Material", sorted(df_rm["Item_Description"].unique()))

    status = st.multiselect(
        "Status",
        sorted(df_rm["QA_Status"].unique()),
        default=df_rm["QA_Status"].unique(),
    )

    df_rm = df_rm.query("QA_Status in @status and Item_Description == @material").copy()

    path = (
        r"L:\Projects\Raw_Materials_Management_Review_Initiative"
        + r"\General\Mgmt Review Log Message Key.xlsx"
    )

    log_message_key = pd.read_excel(path)

    df_rm["Log_Message"] = df_rm["Log_Message"].str.lower()

    df_rm = df_rm.merge(
        right=log_message_key,
        left_on="Log_Message",
        right_on="Log_Message",
        how="left",
    )

    df_rm_summary = (
        df_rm.groupby(["Item_Description", "Alias"], as_index=False)
        .agg({"LOT_NUMBER": "count", "Qty_kg": "sum"})
        .rename(columns={"LOT_NUMBER": "Bag_Count", "Qty_kg": "Quantity_kg"})
        .sort_values("Quantity_kg")
    )

    sns.set_context("paper")

    fig = sns.catplot(
        data=df_rm_summary,
        x="Alias",
        y="Bag_Count",
        kind="bar",
    )

    plt.title(material)
    plt.xlabel("Log_Message")
    plt.xticks(rotation=90)

    ax = fig.facet_axis(0, 0)
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height() + p.get_width() / 2,
            "{0:.0f}".format(p.get_height()),
            color="black",
            rotation="horizontal",
            size="large",
            ha="center",
        )

    plt.tight_layout()

    st.pyplot(fig)

    now = dt.datetime.now().strftime("%y%m%d")
    output = uniquify(f"InventoryPlot.{material}.{now}.png")

    img = BytesIO()
    plt.savefig(img, format="png")

    st.download_button(
        label="Download plot as png",
        data=img,
        file_name=output,
        mime="image/png",
    )

    return material


def run_tab2(df_rm, df_prof):
    color_labels, psd_labels, color_headers, psd_headers = find_headers(df_prof)

    df_prof["Bag_Numbers"] = pd.to_numeric(df_prof["Bag_Numbers"], errors="coerce")

    df_color = clean_proficient_data(df_prof, color_headers)
    df_psd = clean_proficient_data(df_prof, psd_headers)

    df_prof_color_fill_blanks = calc_lot_averages(
        df_rm, df_color, color_headers, color_labels
    )
    df_prof_psd_fill_blanks = calc_lot_averages(df_rm, df_psd, psd_headers, psd_labels)

    df_final = merge_tests_and_averages(
        df_rm, df_color, df_prof_color_fill_blanks, color_labels, "Color"
    )
    df_final = merge_tests_and_averages(
        df_final, df_psd, df_prof_psd_fill_blanks, psd_labels, "PSD"
    )

    df_final = df_final.drop(
        ["LOT_NUMBER", "PO_or_BOL_x", "Bag_Numbers_x", "PO_or_BOL_y", "Bag_Numbers_y"],
        axis=1,
    )

    now = dt.datetime.now().strftime("%y%m%d")
    output = f"PROFxINV.{now}.csv"
    output = uniquify(output)

    st.dataframe(df_final)

    st.download_button(
        label="Download data as CSV",
        data=convert_df(df_final),
        file_name=output,
        mime="text/csv",
    )

    return df_final


def run_tab3(df_final):
    labels = ["Lot", "Bag", "Location", "Log_Message"]
    material_type = st.selectbox("Material Type", ["Grit", "Powder"])
    test = st.selectbox("QA Test", ["Color", "PSD"])
    log_message = st.multiselect(
        "Log_Message",
        sorted(df_final["Log_Message"].unique()),
        default=df_final["Log_Message"].unique(),
    )

    set_max_dist_limit = st.checkbox("Set Max Distance Limit")

    if set_max_dist_limit:
        max_dist_limit = st.number_input("Max Distance", value=0.5)

    if st.checkbox("Calculate", key=1):

        df_clusters = df_final[df_final["Log_Message"].isin(log_message)].copy()

        final_labels = labels.copy()

        parameters = find_parameters(material_type, test)

        final_labels.append(f"{test}_Result_Source")
        final_labels.extend(parameters)

        df_clusters = df_clusters[final_labels].dropna()
        test_df = df_clusters[parameters]

        inertias = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, n_init="auto")
            kmeans.fit(test_df)
            inertias.append(kmeans.inertia_)

        number_of_clusters = KneeLocator(
            range(1, 11),
            inertias,
            curve="convex",
            direction="decreasing",
        ).elbow

        max_dist_from_centroid = 100
        if set_max_dist_limit:
            while max_dist_from_centroid > max_dist_limit:

                clustering = KMeans(n_clusters=number_of_clusters, n_init="auto").fit(
                    test_df
                )

                cluster_centers = clustering.cluster_centers_
                classes = clustering.labels_

                df_clusters["Cluster"] = classes
                for i, x in enumerate(parameters):
                    df_clusters[f"Cluster_Center_{x}"] = [
                        cluster_centers[j][i] for j in classes
                    ]

                df_clusters["Distance_from_Cluster_Center"] = [
                    distance.euclidean(
                        df_clusters[parameters].values.tolist()[i],
                        cluster_centers[j],
                    )
                    for i, j in enumerate(classes)
                ]

                max_dist_from_centroid = df_clusters[
                    "Distance_from_Cluster_Center"
                ].max()
                number_of_clusters += 1

        clustering = KMeans(n_clusters=number_of_clusters, n_init="auto").fit(test_df)

        cluster_centers = clustering.cluster_centers_
        classes = clustering.labels_

        df_clusters["Cluster"] = classes
        for i, x in enumerate(parameters):
            df_clusters[f"Cluster Center - {x}"] = [
                cluster_centers[j][i] for j in classes
            ]

        df_clusters["Distance_from_Cluster_Center"] = [
            distance.euclidean(
                df_clusters[parameters].values.tolist()[i],
                cluster_centers[j],
            )
            for i, j in enumerate(classes)
        ]

        plot_args = {"x": "Cluster", "y": "size", "kind": "bar"}

        if test == "Color":
            df_clusters["Within_Limit"] = (
                df_clusters["Distance_from_Cluster_Center"] <= 0.50
            )

            plot_args["hue"] = "Within_Limit"
            plot_args["hue_order"] = [True, False]
            plot_args["palette"] = ["C2", "C3"]

            cluster_counts = df_clusters.groupby(
                ["Cluster", "Within_Limit"], as_index=False
            ).size()

        else:
            cluster_counts = df_clusters.groupby("Cluster", as_index=False).size()

        plot_args["data"] = cluster_counts

        st.metric("Clusters", len(df_clusters["Cluster"].unique()))

        fig = sns.catplot(**plot_args)

        ax = fig.facet_axis(0, 0)
        for p in ax.patches:
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + p.get_width() / 2,
                "{0:.0f}".format(p.get_height()),
                color="black",
                rotation="horizontal",
                size="large",
                ha="center",
            )

        st.pyplot(fig)

        now = dt.datetime.now().strftime("%y%m%d")
        output = f"Clusters.{now}.csv"
        output = uniquify(output)

        st.download_button(
            label="Download data as CSV",
            data=convert_df(df_clusters),
            file_name=output,
            mime="text/csv",
        )

    return df_clusters, parameters


def run_tab4(df_clusters, parameters, material):
    x = st.selectbox("X", parameters, index=0)
    y = st.selectbox("Y", parameters, index=1)
    z = st.selectbox("Z", parameters, index=2)

    fig = px.scatter_3d(
        data_frame=df_clusters,
        x=x,
        y=y,
        z=z,
        color="Cluster",
        title=material,
    )

    make_spec_box = st.checkbox("Specification Box")
    if make_spec_box:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(f"{x}")
            x_specs = [
                st.number_input(label="Upper", key=f"{x} Upper"),
                st.number_input(label="Lower", key=f"{x} Lower"),
            ]
        with col2:
            st.subheader(f"{y}")
            y_specs = [
                st.number_input(label="Upper", key=f"{y} Upper"),
                st.number_input(label="Lower", key=f"{y} Lower"),
            ]
        with col3:
            st.subheader(f"{z}")
            z_specs = [
                st.number_input(label="Upper", key=f"{z} Upper"),
                st.number_input(label="Lower", key=f"{z} Lower"),
            ]

        cube_data = {
            "x": [
                x_specs[0],
                x_specs[1],
                x_specs[0],
                x_specs[1],
                x_specs[0],
                x_specs[1],
                x_specs[0],
                x_specs[1],
            ],
            "y": [
                y_specs[1],
                y_specs[1],
                y_specs[0],
                y_specs[0],
                y_specs[1],
                y_specs[1],
                y_specs[0],
                y_specs[0],
            ],
            "z": [
                z_specs[1],
                z_specs[1],
                z_specs[1],
                z_specs[1],
                z_specs[0],
                z_specs[0],
                z_specs[0],
                z_specs[0],
            ],
            "opacity": 0.1,
            "color": "green",
            "alphahull": 1,
            "hoverinfo": "skip",
            "name": "Spec Limit",
            "hovertemplate": "b=%{x:.2f}<br>" + "a=%{y:.2f}<br>" + "L=%{z:.2f}",
        }

        fig.add_trace(go.Mesh3d(cube_data))

    fig.update_scenes(xaxis_autorange="reversed")

    if st.checkbox("Calculate", key=2):
        st.plotly_chart(fig, theme="streamlit")

        # download_plot = st.button("Download plot as HTML")
        # if download_plot:
        #     now = dt.datetime.now().strftime("%y%m%d")
        #     output = f"ClustersPlot.{now}.html"
        #     output = uniquify(output)
        #     plotly.offline.plot(fig, filename=output)


def main():

    path1 = st.sidebar.file_uploader(
        "Upload RM Inventory data", type="csv", accept_multiple_files=False
    )

    path_3 = st.sidebar.file_uploader(
        "Upload Proficient data", type="txt", accept_multiple_files=False
    )

    tab_names = [
        "Status-Reason Breakdown",
        "Proficient & Inventory Merger",
        "Data Clustering",
        "3D Plotting of Cluster",
    ]

    tab1, tab2, tab3, tab4 = st.tabs(tab_names)

    with tab1:
        if path1:

            df_rm = pd.read_csv(path1, thousands=",")

            df_rm = format_headers(df_rm)

            material = run_tab1(df_rm)

        elif not path_3:
            st.warning("Upload RM Inventory data.")

    with tab2:
        if path1 and path_3:
            df_prof = pd.read_csv(
                path_3, sep="\t", parse_dates=["Date"], quoting=csv.QUOTE_NONE
            )

            df_prof = format_headers(df_prof)

            df_final = run_tab2(df_rm, df_prof)

        elif not path1 and not path_3:
            st.warning("Upload RM Inventory and Proficient data.")
        elif not path1:
            st.warning("Upload RM Inventory data.")
        elif not path_3:
            st.warning("Upload Proficient data.")

    with tab3:
        if path1 and path_3:
            try:
                df_clusters, parameters = run_tab3(df_final)

            except Exception as e:
                st.error(e)

        elif not path1 and not path_3:
            st.warning("Upload RM Inventory and Proficient data.")

        elif not path1:
            st.warning("Upload RM Inventory data.")

        elif not path_3:
            st.warning("Upload Proficient data.")

    with tab4:
        if path1 and path_3:
            try:
                run_tab4(df_clusters, parameters, material)

            except Exception as e:
                st.error(e)

        elif not path1 and not path_3:
            st.warning("Upload RM Inventory and Proficient data.")
        elif not path1:
            st.warning("Upload RM Inventory data.")
        elif not path_3:
            st.warning("Upload Proficient data.")


if __name__ == "__main__":
    main()
