import streamlit as st
import pandas as pd
import tempfile
import os
from rank_candidates import rank_candidates

st.set_page_config(
    page_title="AI Candidate Ranking System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Candidate Ranking System")

st.write(
    """
    Upload candidate data and a job description.
    The system will rank candidates and generate
    a submission CSV.
    """
)

candidate_file = st.file_uploader(
    "Upload candidates",
    type=["json","jsonl"]
)

jd_file = st.file_uploader(
    "Upload job description",
    type=["docx"]
)

if st.button("Run Ranking"):

    if candidate_file is None:
        st.error("Please upload candidates.jsonl")
        st.stop()

    if jd_file is None:
        st.error("Please upload job_description.docx")
        st.stop()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jsonl"
    ) as temp_candidates:

        temp_candidates.write(
            candidate_file.getbuffer()
        )

        candidate_path = temp_candidates.name

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".docx"
    ) as temp_jd:

        temp_jd.write(
            jd_file.getbuffer()
        )

        jd_path = temp_jd.name

    try:

        with st.spinner(
            "Ranking candidates..."
        ):

            submission = rank_candidates(
                candidate_path,
                jd_path
            )

        st.success(
            "Ranking completed successfully!"
        )

        st.subheader(
            "Top Ranked Candidates"
        )

        st.dataframe(
            submission,
            use_container_width=True
        )

        csv_data = submission.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download submission.csv",
            data=csv_data,
            file_name="submission.csv",
            mime="text/csv"
        )
            except Exception as e:
        import traceback

        st.error(str(e))
        st.code(traceback.format_exc())

    finally:

        if os.path.exists(candidate_path):
            os.remove(candidate_path)

        if os.path.exists(jd_path):
            os.remove(jd_path)

    finally:

        if os.path.exists(candidate_path):
            os.remove(candidate_path)

        if os.path.exists(jd_path):
            os.remove(jd_path)
