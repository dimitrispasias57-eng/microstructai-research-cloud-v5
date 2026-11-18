"""Simple Streamlit presentation of existing metrics and their QC provenance."""

import pandas as pd
import streamlit as st

from .qc_plotting import available_metrics, count_caption, metric_view, plot_qc_metric, qc_details, view_counts
from .ui_cache import figure_png, frame_key, remembered_view


def render_qc_plots(df: pd.DataFrame, x_axis: str) -> None:
    st.subheader("Results over time")
    st.caption("Colours show accepted results. Gaps mean excluded or unavailable values—not zero. "
               "These display controls do not change your analysis settings, scores or saved results.")
    controls = st.columns(2)
    show_excluded = controls[0].checkbox(
        "Show excluded values in grey", value=False, key="qc_plots_show_excluded",
        help="Optional unconnected raw points for troubleshooting. They are never added to accepted curves or statistics.",
    )
    show_strip = controls[1].checkbox(
        "Show QC strip below plots", value=True, key="qc_plots_show_strip",
        help="Shows the location of every frame, even when its metric cannot be calculated.",
    )
    st.caption("● Accepted · ▲ Warning, still included · grey × in QC strip: excluded · "
               "red | in QC strip: unavailable · grey square/dotted curve: no metric-specific QC")
    if show_excluded:
        st.info("Grey circles are raw diagnostic values. QC scores can use a different stored normalization, "
                "so a grey score is not an alternative trusted estimate. Grey points do not affect statistics.")
    groups = [
        ("Scores", {"score"}, {"SASI", "AggIndex", "GapScore", "PTI"}),
        ("Gap structure", {"mask", "shape"}, {"Xi", "diameter", "area", "count"}),
        ("Optical & texture", {"image"}, {"optical_mean", "optical_cv"}),
    ]
    selected_view = remembered_view("Plot category", [group[0] for group in groups] + ["QC details & values"],
                                    "qc_plot_category")
    data_key = frame_key(df)
    for name, families, defaults in groups:
        if selected_view == name:
            specs = available_metrics(df, families)
            if not specs:
                st.info("These metrics were not available in this analysis.")
                continue
            if name == "Scores":
                st.caption("Stored QC-aware research scores, restricted to the selected main-summary frames. "
                           "These are not validated biomarkers or commercial-instrument aggregation indices. "
                           "Normalization is within-sequence unless your analysis used a locked reference.")
            elif name == "Gap structure":
                st.caption("Bright plasma-gap geometry—not individual RBC measurements. "
                           "Shape-dependent measurements also respect the existing sparse-mask flags.")
            else:
                st.info("Optical/texture measurements are assessed separately from mask geometry. "
                        "A mask failure alone does not discard intensity values. When optical QC is unavailable, "
                        "the measured curve is dotted grey and is not described as QC-accepted.")
            by_key = {spec.key: spec for spec in specs}
            selected = st.multiselect(
                "Metrics to show", list(by_key),
                default=[spec.key for spec in specs if spec.key in defaults],
                format_func=lambda key, labels=by_key: labels[key].label, key=f"qc_plots_metrics_{name}",
            )
            columns = st.columns(2)
            for index, key in enumerate(selected):
                spec = by_key[key]
                with columns[index % 2]:
                    png = figure_png(("qc", data_key, spec.key, x_axis, show_excluded, show_strip),
                                     lambda: plot_qc_metric(df, spec, x_axis, show_excluded, show_strip))
                    st.image(png, width="stretch")
                    view = metric_view(df, spec)
                    st.caption(count_caption(view))
                    st.caption(f"Value source: {view['source_column'].iloc[0]}")
            if not selected:
                st.info("Choose a metric above to display its curve.")
    if selected_view == "QC details & values":
        specs = available_metrics(df)
        rows = []
        for spec in specs:
            view = metric_view(df, spec)
            accepted = view.loc[view["status"].isin(["Accepted", "Warning (accepted)"]), "value"]
            rows.append({"Metric": spec.label, **view_counts(view), "Accepted mean": accepted.mean(),
                         "Accepted median": accepted.median(), "Value source": view["source_column"].iloc[0]})
        st.caption("Warnings are included in the accepted count. Means/medians use only accepted values for each metric; "
                   "excluded, unavailable and unassessed values never enter these statistics. "
                   "Frame counts are not independent donor counts.")
        counts = pd.DataFrame(rows)
        st.dataframe(counts, width="stretch", hide_index=True)
        if specs:
            selected_key = st.selectbox("Inspect a metric", [spec.key for spec in specs],
                                        format_func=lambda key: next(spec.label for spec in specs if spec.key == key),
                                        key="qc_plots_detail_metric")
            detail = qc_details(df, selected_key)
            st.dataframe(detail, width="stretch", hide_index=True)
            st.download_button("Download this plot's values and QC reasons", detail.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"plot_QC_{selected_key}.csv", mime="text/csv", key="qc_plots_detail_download")
        st.caption("Original per-frame exports remain intact. No missing value is replaced with zero or interpolated.")
