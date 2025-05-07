import streamlit as st
from utils.langchain_setup import create_vectorstore, get_qa_chain

st.set_page_config(page_title="Product Manual Chatbot", layout="wide")

# Define products with display names and PDF paths
PRODUCTS = {
    "Measuretrol": "manuals/UM-Measuretrol.pdf",
    "Qualsteam": "manuals/UM-QualSteam_R7_16072023.pdf",
    "Aqua2Trans (Cond./TDS)": "manuals/Aqua2Trans Conductivity - TDS User Manual R1 26Mar2025.pdf",
    "Aqua2Trans (pH/ORP)": "manuals/Aqua2Trans pH - ORP User Manual R1_26Mar2025.pdf",
    "Aqua4Trans": "manuals/Aqua4Trans User Manual R2 Dec 2023.pdf",
    "CX2000": "manuals/Multiparameter Analyser CX2000 Users Manual_R0_190423.pdf",
    "RTrU301" : "manuals/Operational Manual_07_12_2021.pdf",
}

# Initialize session state
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None
    st.session_state.chat_history = []
    st.session_state.qa_chain = None
    st.session_state.vectorstore = None

# Sidebar content
def show_sidebar():
    st.sidebar.image("fm-logo.png", use_container_width=True)
    st.sidebar.title("Product Manual Chatbot")
    st.sidebar.markdown("### Welcome to the Product Manual Chatbot!")
    st.sidebar.markdown("""
    Select a product from the list to start chatting with its user manual. Once you choose a product, you can ask questions about the manual, and I will help you find answers.
    """)
    st.sidebar.markdown("#### Instructions:")
    st.sidebar.markdown("""
    1. **Select a product**: Choose the product whose manual you want to explore.
    2. **Ask questions**: Once the product is selected, ask me anything about the manual, and I will provide you with relevant information.
    3. **Back to product selection**: You can always go back and choose a different product using the button in the chat space.
    """)

    # Product selection buttons
    st.sidebar.markdown("### Select a product to start:")
    for name, path in PRODUCTS.items():
        if st.sidebar.button(name):
            st.session_state.selected_product = name
            # st.session_state.vectorstore = create_vectorstore(path)
            st.session_state.vectorstore = create_vectorstore(path, name)
            st.session_state.qa_chain = get_qa_chain(
                st.session_state.vectorstore,
                product_name=name
            )
            st.rerun()

# Show product selection screen
def show_home():
    st.title("Product Manual Chatbot")
    st.markdown("### Please select a product from the sidebar to start chatting with its manual.")

# Show chat interface
def show_chat():
    st.title(f"Chat with *{st.session_state.selected_product}* Manual")
    
    if st.button("Back to Product Selection"):
        st.session_state.selected_product = None
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.rerun()

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).markdown(msg)

    prompt = st.chat_input("Ask a question about the product...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append(("user", prompt))
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.run(prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append(("assistant", response))

# Page router
show_sidebar()

if st.session_state.selected_product is None:
    show_home()
else:
    show_chat()
