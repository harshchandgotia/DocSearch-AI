import streamlit as st
import requests

# --- Page Config ---
st.set_page_config(
    page_title="DocSearch AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State ---
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "url_rows" not in st.session_state:
    st.session_state.url_rows = [""]
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "active_session_messages" not in st.session_state:
    st.session_state.active_session_messages = []
if "active_session_pdf_ids" not in st.session_state:
    st.session_state.active_session_pdf_ids = []
if "sessions_list" not in st.session_state:
    st.session_state.sessions_list = []
if "show_new_chat_form" not in st.session_state:
    st.session_state.show_new_chat_form = False


# --- Helpers ---
def get_headers():
    return {"Authorization": f"Bearer {api_key}"}


def api_available():
    return bool(api_key and api_key.strip())


def fetch_sessions():
    if not api_available():
        return
    try:
        resp = requests.get(f"{api_url}/sessions", headers=get_headers(), timeout=10)
        if resp.status_code == 200:
            st.session_state.sessions_list = resp.json().get("sessions", [])
    except Exception:
        pass


def load_session_messages(session_id):
    if not api_available():
        return
    try:
        resp = requests.get(
            f"{api_url}/sessions/{session_id}/messages",
            headers=get_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            st.session_state.active_session_messages = resp.json().get("messages", [])
    except Exception:
        pass


def get_support_color(level):
    if level == "fully_supported":
        return "green"
    elif level == "partially_supported":
        return "orange"
    elif level == "not_supported":
        return "red"
    return "gray"


# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
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
    st.subheader("Indexed Documents")

    if st.session_state.documents:
        for pdf_id, filename in list(st.session_state.documents.items()):
            col_name, col_btn = st.columns([4, 1])
            with col_name:
                st.markdown(f"**{filename}**  \n`{pdf_id[:8]}...`")
            with col_btn:
                if st.button("X", key=f"del_{pdf_id}", help=f"Delete {filename}"):
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
        if st.button("Clear Session", type="secondary", use_container_width=True):
            st.session_state.documents = {}
            st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.subheader("Chat Sessions")

    # Fetch sessions on each render
    fetch_sessions()

    if st.button("+ New Chat", type="primary", use_container_width=True):
        st.session_state.show_new_chat_form = True

    # New chat form
    if st.session_state.show_new_chat_form:
        if not st.session_state.documents:
            st.warning("Upload documents first before creating a chat session.")
            st.session_state.show_new_chat_form = False
        else:
            doc_labels = [
                f"{fname} ({pid[:8]}...)"
                for pid, fname in st.session_state.documents.items()
            ]
            doc_ids = list(st.session_state.documents.keys())

            selected_docs = st.multiselect(
                "Select documents for this session",
                options=doc_labels,
                default=doc_labels,
                key="new_chat_docs",
            )

            if st.button("Create Session", key="create_session_btn"):
                if not selected_docs:
                    st.error("Select at least one document.")
                elif api_available():
                    selected_pdf_ids = [
                        doc_ids[doc_labels.index(label)] for label in selected_docs
                    ]
                    try:
                        resp = requests.post(
                            f"{api_url}/sessions",
                            json={"pinned_pdf_ids": selected_pdf_ids},
                            headers=get_headers(),
                            timeout=10,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.active_session_id = data["session_id"]
                            st.session_state.active_session_pdf_ids = data["pinned_pdf_ids"]
                            st.session_state.active_session_messages = []
                            st.session_state.show_new_chat_form = False
                            st.rerun()
                        else:
                            st.error(f"Failed: {resp.json().get('error', 'unknown')}")
                    except Exception as exc:
                        st.error(f"Error creating session: {exc}")

            if st.button("Cancel", key="cancel_new_chat"):
                st.session_state.show_new_chat_form = False
                st.rerun()

    # List existing sessions
    for session in st.session_state.sessions_list:
        sid = session["session_id"]
        title = session.get("title") or "Untitled"
        msg_count = session.get("message_count", 0)
        is_active = sid == st.session_state.active_session_id

        col_sess, col_del = st.columns([5, 1])
        with col_sess:
            label = f"**{title[:30]}**" if is_active else title[:30]
            if st.button(
                f"{label} ({msg_count} msgs)",
                key=f"sess_{sid}",
                use_container_width=True,
            ):
                st.session_state.active_session_id = sid
                st.session_state.active_session_pdf_ids = session.get("pinned_pdf_ids", [])
                load_session_messages(sid)
                st.session_state.show_new_chat_form = False
                st.rerun()
        with col_del:
            if st.button("X", key=f"del_sess_{sid}", help="Delete session"):
                if api_available():
                    try:
                        requests.delete(
                            f"{api_url}/sessions/{sid}",
                            headers=get_headers(),
                            timeout=10,
                        )
                        if sid == st.session_state.active_session_id:
                            st.session_state.active_session_id = None
                            st.session_state.active_session_messages = []
                            st.session_state.active_session_pdf_ids = []
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")


# --- Header ---
st.title("DocSearch AI")
st.caption("Upload PDFs and ask natural-language questions about their content.")
st.divider()

# --- Tabs ---
tab_upload, tab_chat = st.tabs(["Upload Documents", "Chat"])


# == Upload Tab ==
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
            if len(st.session_state.url_rows) > 1 and st.button("x", key=f"rm_url_{i}"):
                rows_to_remove.append(i)

    if rows_to_remove:
        for idx in sorted(rows_to_remove, reverse=True):
            st.session_state.url_rows.pop(idx)
        st.rerun()

    if st.button("+ Add URL"):
        st.session_state.url_rows.append("")
        st.rerun()

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
                with st.spinner(f"Uploading {len(uploaded_files)} file(s)..."):
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
                            errors.append("File upload: authentication failed -- check your API key.")
                        else:
                            errors.append(f"File upload error: {resp.json().get('error', 'unknown')}")
                    except requests.exceptions.ConnectionError:
                        errors.append(f"Could not connect to {api_url}. Is the server running?")
                    except requests.exceptions.Timeout:
                        errors.append("File upload timed out. Large PDFs can take a few minutes.")
                    except Exception as exc:
                        errors.append(f"Unexpected error during file upload: {exc}")

            if has_urls:
                with st.spinner(f"Indexing {len(valid_urls)} URL(s)..."):
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
                            errors.append("URL upload: authentication failed -- check your API key.")
                        else:
                            errors.append(f"URL upload error: {resp.json().get('error', 'unknown')}")
                    except requests.exceptions.ConnectionError:
                        errors.append(f"Could not connect to {api_url}. Is the server running?")
                    except requests.exceptions.Timeout:
                        errors.append("URL indexing timed out.")
                    except Exception as exc:
                        errors.append(f"Unexpected error during URL indexing: {exc}")

            if combined_results:
                st.session_state.documents.update(combined_results)
                st.success(f"Indexed {len(combined_results)} document(s) successfully.")
                for pid, fname in combined_results.items():
                    st.markdown(f"- **{fname}** -> `{pid}`")

            for err in errors:
                st.error(err)


# == Chat Tab ==
with tab_chat:
    if not st.session_state.active_session_id:
        st.info("Select or create a chat session from the sidebar to get started.")
    else:
        session_id = st.session_state.active_session_id
        pinned_ids = st.session_state.active_session_pdf_ids

        # Show session info
        pinned_names = [
            st.session_state.documents.get(pid, pid[:8] + "...")
            for pid in pinned_ids
        ]
        st.caption(f"Session documents: {', '.join(pinned_names)}")
        st.divider()

        # Display chat history
        for msg in st.session_state.active_session_messages:
            role = msg["role"]
            content = msg["content"]

            with st.chat_message(role):
                st.markdown(content)

                # Show metadata for assistant messages
                if role == "assistant" and msg.get("is_supported") is not None:
                    with st.expander("Details"):
                        support_level = msg.get("is_supported", "")
                        color = get_support_color(support_level)
                        st.markdown(f"**Support:** :{color}[{support_level}]")

                        usefulness = msg.get("is_useful", "")
                        if usefulness:
                            st.markdown(f"**Usefulness:** {usefulness}")

                        rev_count = msg.get("revision_count")
                        rew_count = msg.get("rewrite_count")
                        if rev_count is not None:
                            st.markdown(f"**Revisions:** {rev_count}")
                        if rew_count is not None:
                            st.markdown(f"**Query rewrites:** {rew_count}")

                        sources = msg.get("sources", [])
                        if sources:
                            source_names = [
                                st.session_state.documents.get(s, s[:8] + "...")
                                for s in sources
                            ]
                            st.markdown(f"**Sources:** {', '.join(source_names)}")

                        retrieval_used = msg.get("retrieval_used")
                        if retrieval_used is not None:
                            st.markdown(f"**Retrieval used:** {'Yes' if retrieval_used else 'No'}")

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not api_available():
                st.error("Set your API key in the sidebar first.")
            else:
                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Add to local state
                st.session_state.active_session_messages.append({
                    "role": "user",
                    "content": prompt,
                })

                # Call the query API
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            resp = requests.post(
                                f"{api_url}/query",
                                json={
                                    "question": prompt,
                                    "pinned_pdf_ids": pinned_ids,
                                    "session_id": session_id,
                                },
                                headers=get_headers(),
                                timeout=120,
                            )

                            if resp.status_code == 200:
                                data = resp.json()
                                answer = data.get("answer", "No answer returned.")
                                st.markdown(answer)

                                # Show details expander
                                with st.expander("Details"):
                                    support_level = data.get("is_supported", "")
                                    if support_level:
                                        color = get_support_color(support_level)
                                        st.markdown(f"**Support:** :{color}[{support_level}]")

                                    usefulness = data.get("is_useful", "")
                                    if usefulness:
                                        st.markdown(f"**Usefulness:** {usefulness}")

                                    st.markdown(f"**Revisions:** {data.get('revision_count', 0)}")
                                    st.markdown(f"**Query rewrites:** {data.get('rewrite_count', 0)}")

                                    sources = data.get("sources", [])
                                    if sources:
                                        source_names = [
                                            st.session_state.documents.get(s, s[:8] + "...")
                                            for s in sources
                                        ]
                                        st.markdown(f"**Sources:** {', '.join(source_names)}")

                                    st.markdown(f"**Retrieval used:** {'Yes' if data.get('retrieval_used') else 'No'}")

                                # Add assistant message to local state
                                st.session_state.active_session_messages.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "is_supported": data.get("is_supported"),
                                    "is_useful": data.get("is_useful"),
                                    "revision_count": data.get("revision_count"),
                                    "rewrite_count": data.get("rewrite_count"),
                                    "sources": data.get("sources"),
                                    "retrieval_used": data.get("retrieval_used"),
                                })

                            elif resp.status_code == 401:
                                st.error("Authentication failed -- check your API key.")
                            else:
                                msg = resp.json().get("error", "Unknown server error.")
                                st.error(f"Server error: {msg}")

                        except requests.exceptions.ConnectionError:
                            st.error(f"Could not connect to {api_url}. Is the FastAPI server running?")
                        except requests.exceptions.Timeout:
                            st.error("Request timed out.")
                        except Exception as exc:
                            st.error(f"Unexpected error: {exc}")
