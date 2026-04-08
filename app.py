import streamlit as st
import requests
from collections import defaultdict

# --- Page Config ---
st.set_page_config(
    page_title="DocSearch AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State ---
if "documents" not in st.session_state:
    st.session_state.documents = {}       # {pdf_id: filename}
if "url_rows" not in st.session_state:
    st.session_state.url_rows = [""]      # list of URL strings for dynamic rows
if "question_rows" not in st.session_state:
    st.session_state.question_rows = [{"question": "", "pdf_id": None}]


# --- Helpers ---
def get_headers():
    return {"Authorization": f"Bearer {api_key}"}

def api_available():
    return bool(api_key and api_key.strip())


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="Base URL of the running DocSearch API server",
    )
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Bearer token set in your .env file (API_KEY)",
    )

    st.divider()
    st.subheader("📂 Indexed Documents")

    if st.session_state.documents:
        for pdf_id, filename in list(st.session_state.documents.items()):
            col_name, col_btn = st.columns([4, 1])
            with col_name:
                st.markdown(f"**{filename}**  \n`{pdf_id[:8]}…`")
            with col_btn:
                if st.button("🗑️", key=f"del_{pdf_id}", help=f"Delete {filename}"):
                    if api_available():
                        try:
                            resp = requests.delete(
                                f"{api_url}/documents/{pdf_id}",
                                headers=get_headers(),
                                timeout=30,
                            )
                            if resp.status_code == 200:
                                del st.session_state.documents[pdf_id]
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {resp.json().get('error', 'unknown')}")
                        except Exception as exc:
                            st.error(f"Error: {exc}")
                    else:
                        st.warning("Set API key first.")

        st.write("")
        if st.button("🗑️ Clear Session", type="secondary", use_container_width=True):
            st.session_state.documents = {}
            st.rerun()
    else:
        st.caption("No documents indexed yet.")


# --- Header ---
st.title("🔍 DocSearch AI")
st.caption("Upload PDFs and ask natural-language questions about their content.")
st.divider()

# --- Tabs ---
tab_upload, tab_query = st.tabs(["📤  Upload Documents", "💬  Ask Questions"])


# ── Upload Tab ─────────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Index PDF Documents")

    # Section 1: File upload
    st.markdown("#### Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Section 2: URL inputs
    st.markdown("#### Add URLs")

    rows_to_remove = []
    for i, url_val in enumerate(st.session_state.url_rows):
        col_input, col_remove = st.columns([9, 1])
        with col_input:
            st.session_state.url_rows[i] = st.text_input(
                f"URL {i + 1}",
                value=url_val,
                key=f"url_{i}",
                label_visibility="collapsed",
                placeholder="https://example.com/document.pdf",
            )
        with col_remove:
            if len(st.session_state.url_rows) > 1 and st.button("×", key=f"rm_url_{i}"):
                rows_to_remove.append(i)

    # Apply removals after the render loop to avoid index shifting mid-loop
    if rows_to_remove:
        for idx in sorted(rows_to_remove, reverse=True):
            st.session_state.url_rows.pop(idx)
        st.rerun()

    if st.button("+ Add URL"):
        st.session_state.url_rows.append("")
        st.rerun()

    # Combined upload button
    st.write("")
    upload_clicked = st.button(
        "Upload & Index",
        type="primary",
        disabled=not api_available(),
    )

    if not api_available():
        st.info("Enter your API key in the sidebar to get started.")

    if upload_clicked:
        valid_urls = [u.strip() for u in st.session_state.url_rows if u.strip()]
        has_files = bool(uploaded_files)
        has_urls = bool(valid_urls)

        if not has_files and not has_urls:
            st.error("Please upload at least one file or enter at least one URL.")
        else:
            combined_results = {}
            errors = []

            if has_files:
                with st.spinner(f"Uploading {len(uploaded_files)} file(s)…"):
                    try:
                        files_payload = [
                            ("files", (f.name, f.read(), "application/pdf"))
                            for f in uploaded_files
                        ]
                        resp = requests.post(
                            f"{api_url}/upload",
                            files=files_payload,
                            headers=get_headers(),
                            timeout=300,
                        )
                        if resp.status_code == 200:
                            combined_results.update(resp.json().get("Files uploaded", {}))
                        elif resp.status_code == 401:
                            errors.append("File upload: authentication failed — check your API key.")
                        else:
                            errors.append(f"File upload error: {resp.json().get('error', 'unknown')}")
                    except requests.exceptions.ConnectionError:
                        errors.append(f"Could not connect to `{api_url}`. Is the server running?")
                    except requests.exceptions.Timeout:
                        errors.append("File upload timed out. Large PDFs can take a few minutes.")
                    except Exception as exc:
                        errors.append(f"Unexpected error during file upload: {exc}")

            if has_urls:
                with st.spinner(f"Indexing {len(valid_urls)} URL(s)…"):
                    try:
                        resp = requests.post(
                            f"{api_url}/extract",
                            json={"documents": valid_urls},
                            headers=get_headers(),
                            timeout=300,
                        )
                        if resp.status_code == 200:
                            combined_results.update(resp.json().get("Files uploaded", {}))
                        elif resp.status_code == 401:
                            errors.append("URL upload: authentication failed — check your API key.")
                        else:
                            errors.append(f"URL upload error: {resp.json().get('error', 'unknown')}")
                    except requests.exceptions.ConnectionError:
                        errors.append(f"Could not connect to `{api_url}`. Is the server running?")
                    except requests.exceptions.Timeout:
                        errors.append("URL indexing timed out.")
                    except Exception as exc:
                        errors.append(f"Unexpected error during URL indexing: {exc}")

            if combined_results:
                st.session_state.documents.update(combined_results)
                st.success(f"Indexed {len(combined_results)} document(s) successfully.")
                for pid, fname in combined_results.items():
                    st.markdown(f"- **{fname}** → `{pid}`")

            for err in errors:
                st.error(err)


# ── Query Tab ──────────────────────────────────────────────────────────────────
with tab_query:
    st.subheader("Ask Questions")

    if not st.session_state.documents:
        st.info(
            "No documents indexed yet. "
            "Head to the **Upload Documents** tab to add some PDFs first."
        )
    else:
        # Build filename → pdf_id lookup
        doc_options = {
            filename: pdf_id
            for pdf_id, filename in st.session_state.documents.items()
        }
        doc_labels = list(doc_options.keys())
        default_pdf_id = list(doc_options.values())[0]

        # Normalise rows: reset any pdf_id that no longer exists (e.g. after a delete)
        for row in st.session_state.question_rows:
            if row["pdf_id"] not in doc_options.values():
                row["pdf_id"] = default_pdf_id

        # Render question rows
        rows_to_remove_q = []
        for i, row in enumerate(st.session_state.question_rows):
            col_q, col_doc, col_rm = st.columns([5, 2, 0.4])

            with col_q:
                st.session_state.question_rows[i]["question"] = st.text_input(
                    f"Question {i + 1}",
                    value=row["question"],
                    key=f"q_{i}",
                    label_visibility="collapsed",
                    placeholder="Ask a question about the selected document…",
                )

            with col_doc:
                current_label = next(
                    (lbl for lbl, pid in doc_options.items() if pid == row["pdf_id"]),
                    doc_labels[0],
                )
                selected_label = st.selectbox(
                    f"Doc {i + 1}",
                    options=doc_labels,
                    index=doc_labels.index(current_label),
                    key=f"doc_{i}",
                    label_visibility="collapsed",
                )
                st.session_state.question_rows[i]["pdf_id"] = doc_options[selected_label]

            with col_rm:
                if len(st.session_state.question_rows) > 1 and st.button("×", key=f"rm_q_{i}"):
                    rows_to_remove_q.append(i)

        if rows_to_remove_q:
            for idx in sorted(rows_to_remove_q, reverse=True):
                st.session_state.question_rows.pop(idx)
            st.rerun()

        if st.button("+ Add Question"):
            st.session_state.question_rows.append({"question": "", "pdf_id": default_pdf_id})
            st.rerun()

        st.write("")
        ask_clicked = st.button("Ask", type="primary", disabled=not api_available())

        if not api_available():
            st.info("Enter your API key in the sidebar to enable queries.")

        if ask_clicked:
            valid_rows = [
                (i, row["question"].strip(), row["pdf_id"])
                for i, row in enumerate(st.session_state.question_rows)
                if row["question"].strip()
            ]

            if not valid_rows:
                st.error("Please enter at least one question.")
            else:
                # Group questions by pdf_id, preserving original row indices
                groups = defaultdict(list)   # pdf_id -> [(original_row_idx, question)]
                for orig_idx, question, pdf_id in valid_rows:
                    groups[pdf_id].append((orig_idx, question))

                payload = {
                    "questions": [
                        {
                            "pdf_id": pdf_id,
                            "questions": [q for _, q in rows],
                        }
                        for pdf_id, rows in groups.items()
                    ]
                }

                # Build reverse-lookup: (pdf_id, q_index_within_group) -> original_row_idx
                position_map = {}
                for pdf_id, rows in groups.items():
                    for q_idx, (orig_idx, _) in enumerate(rows):
                        position_map[(pdf_id, q_idx)] = orig_idx

                with st.spinner("Querying documents…"):
                    try:
                        response = requests.post(
                            f"{api_url}/query",
                            json=payload,
                            headers=get_headers(),
                            timeout=120,
                        )

                        if response.status_code == 200:
                            answers_groups = response.json().get("answers", [])

                            # Map answers back to original row positions
                            answers_by_position = {}
                            for group in answers_groups:
                                group_pdf_id = group["pdf_id"]
                                for q_idx, answer in enumerate(group.get("answers", [])):
                                    orig_idx = position_map.get((group_pdf_id, q_idx))
                                    if orig_idx is not None:
                                        answers_by_position[orig_idx] = answer

                            st.divider()
                            st.subheader("Answers")
                            for orig_idx, question, pdf_id in valid_rows:
                                answer = answers_by_position.get(orig_idx, "No answer returned.")
                                doc_name = next(
                                    (fname for fname, pid in doc_options.items() if pid == pdf_id),
                                    pdf_id,
                                )
                                with st.expander(f"Q: {question}  ·  [{doc_name}]", expanded=True):
                                    st.write(answer)

                        elif response.status_code == 401:
                            st.error("Authentication failed — check your API key.")
                        else:
                            msg = response.json().get("error", "Unknown server error.")
                            st.error(f"Server error: {msg}")

                    except requests.exceptions.ConnectionError:
                        st.error(f"Could not connect to `{api_url}`. Is the FastAPI server running?")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. Try with fewer questions.")
                    except Exception as exc:
                        st.error(f"Unexpected error: {exc}")
