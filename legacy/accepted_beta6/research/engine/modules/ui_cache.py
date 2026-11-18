"""Bounded, browser-session-only caches for reproducible UI computations.

No shared/global cache of uploaded research data. Data-dependent keys invalidate
results when values, columns, row order, or selected options change.
"""
from __future__ import annotations

import hashlib
import io
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def frame_key(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(repr((list(df.columns), [str(dtype) for dtype in df.dtypes], df.shape)).encode())
    try:
        digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    except TypeError:
        digest.update(pickle.dumps(df, protocol=5))
    return digest.hexdigest()


def session_value(namespace: str, key, factory, max_entries: int = 8):
    root = st.session_state.setdefault("_v3_display_cache", {})
    cache = root.setdefault(namespace, {})
    if key not in cache:
        value = factory()
        while len(cache) >= max_entries:
            cache.pop(next(iter(cache)))
        cache[key] = value
    return cache[key]


def figure_png(key, factory) -> bytes:
    def create():
        fig = factory()
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
            return buffer.getvalue()
        finally:
            plt.close(fig)
    return session_value("plot_png", key, create, max_entries=20)


def cached_pca(df: pd.DataFrame, metrics: list[str], compute):
    identity_columns = ["filename", "imageIndex", "originalImageIndex_0based", "originalImageIndex_1based",
                        "analysisIndex_0based", "analysisIndex_1based"]
    columns = list(dict.fromkeys(col for col in [*metrics, *identity_columns] if col in df))
    key = (tuple(metrics), frame_key(df[columns]))
    return session_value("pca", key, lambda: compute(df, metrics), max_entries=4)


def remembered_view(label: str, options: list[str], key: str, dropdown: bool = False) -> str:
    """Keep a view choice even when Streamlit removes hidden widget state."""
    saved = st.session_state.setdefault("_v3_view_preferences", {})
    previous = saved.get(key, options[0])
    index = options.index(previous) if previous in options else 0
    if dropdown:
        value = st.selectbox(label, options, index=index, key=key)
    else:
        value = st.radio(label, options, index=index, horizontal=True, key=key)
    saved[key] = value
    return value


@dataclass(frozen=True)
class SessionUpload:
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


def remembered_upload(label: str, types: list[str], key: str, container=None):
    """Retain an auxiliary upload when a lazy view's widget is temporarily hidden."""
    owner = st if container is None else container
    saved = st.session_state.setdefault("_v3_auxiliary_uploads", {})

    def changed():
        uploaded = st.session_state.get(key)
        if uploaded is None:
            saved.pop(key, None)
        else:
            saved[key] = SessionUpload(uploaded.name, uploaded.getvalue())

    uploaded = owner.file_uploader(label, type=types, key=key, on_change=changed)
    if uploaded is not None:
        saved[key] = SessionUpload(uploaded.name, uploaded.getvalue())
    retained = saved.get(key)
    if uploaded is None and retained is not None:
        owner.caption(f"Using retained session file: {retained.name}. Upload another file to replace it.")
        if owner.button("Forget retained file", key=f"forget_{key}"):
            saved.pop(key, None)
            return None
    return retained
