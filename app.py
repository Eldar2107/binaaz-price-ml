# app.py — convenience alias for streamlit_app.py
# Both entry-points are identical:
#   streamlit run app.py
#   streamlit run streamlit_app.py
import runpy
runpy.run_path("streamlit_app.py", run_name="__main__")