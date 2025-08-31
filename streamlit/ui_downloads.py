# _ui_downloads.py
import os, tempfile
import streamlit as st

def add_downloads(*, fig=None, deck=None, basename: str = "view"):
    """Reusable Download buttons for Plotly figures or pydeck maps."""
    cols = st.columns(2 if fig is not None else 1)

    # Plotly (Points / Voxel / Mapbox)
    if fig is not None:
        html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
        cols[0].download_button("⬇️ Interactive HTML", html_bytes, f"{basename}.html", "text/html")
        try:
            png_bytes = fig.to_image(format="png", scale=2)  # needs: pip install -U kaleido
            cols[1].download_button("🖼 PNG image", png_bytes, f"{basename}.png", "image/png")
        except Exception:
            cols[1].caption("PNG export needs `kaleido` (pip install -U kaleido).")

    # pydeck (3D Map)
    if deck is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            tmp_path = tmp.name
        try:
            deck.to_html(tmp_path, open_browser=False)
            with open(tmp_path, "rb") as f:
                deck_html = f.read()
        finally:
            try: os.remove(tmp_path)
            except OSError: pass

        st.download_button("3D map (HTML)", deck_html, f"{basename}.html", "text/html")
