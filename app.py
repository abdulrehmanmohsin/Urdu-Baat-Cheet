import streamlit as st
import torch
import traceback


from assignment2_fromscratch import (
    BPETokenizer,
    UrduChatbotDataset,
    Transformer,
    normalize_urdu_text,
    generate_text
)

# -----------------------------------------------------------------------------
# Streamlit Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Urdu Main Baat-Cheet 🤖",
    layout="centered",
    page_icon="💬"
)

# -----------------------------------------------------------------------------
# Model Loader (with tokenizer.json fallback)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        checkpoint_path = "best_BLEU_model.pt"
        tokenizer_path = "tokenizer.json"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        config = checkpoint.get("config", None)
        vocab_size = checkpoint.get("vocab_size", None)
        tokenizer = checkpoint.get("tokenizer", None)

        # If tokenizer is not stored inside checkpoint, load from JSON
        if tokenizer is None:
            st.info("Loading tokenizer from tokenizer.json...")
            tokenizer = BPETokenizer.load(tokenizer_path)
        config = checkpoint.get("config")
        
        # Fallback if config missing
        if config is None:
            config = {
                "vocab_size": 5000,
                "d_model": 256,
                "num_heads": 2,
                "d_ff": 512,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "max_len": 50,
                "dropout": 0.1
            }
        
        # Build dataset meta tokens (<PAD>, <START>, <END>, <UNK>)
        dataset = UrduChatbotDataset([], tokenizer, config["max_len"])

        # Initialize Transformer model
        model = Transformer(
            vocab_size=len(tokenizer.token_to_id),
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            max_len=config["max_len"],
            dropout=config["dropout"],
            pad_idx=dataset.pad_idx
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()

        return model, tokenizer, dataset

    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'best_BLEU_model.pt' exists.")
        st.stop()

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("""
        Troubleshooting steps:
        1. Ensure all files (model + tokenizer.json) are in the same directory.
        2. Match PyTorch version with your training environment.
        3. Re-run 'save_tokenizer.py' if tokenizer.json is missing.
        """)
        st.code(traceback.format_exc())  # 👈 shows detailed stack trace in Streamlit

        st.stop()


# -----------------------------------------------------------------------------
# Page Header
# -----------------------------------------------------------------------------
st.title("💬 Urdu Main Baat-Cheet")
st.markdown("""
An Urdu conversational chatbot built with a **custom Transformer model**  
trained completely **from scratch** using **Byte Pair Encoding (BPE)**.
""")
st.caption("Developed by **Abdul Rehman Mohsin** and **Daniyal Shafiq**")

# -----------------------------------------------------------------------------
# Load Model and Tokenizer
# -----------------------------------------------------------------------------
model, tokenizer, dataset = load_model()

# -----------------------------------------------------------------------------
# Chat Interface
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------------------------------
# User Input
# -----------------------------------------------------------------------------
if prompt := st.chat_input("اردو میں کچھ لکھیں... (Type something in Urdu)"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate model response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            normalized_input = normalize_text(prompt)

            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                input_text=normalized_input,
                max_len=50,
                temperature=0.8,
                device="cpu",
                greedy=True
            )

            # Show and store response
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
