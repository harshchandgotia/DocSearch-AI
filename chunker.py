from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load example documents
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=400,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)