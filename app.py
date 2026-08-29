# app.py

import streamlit as st

from farming_kb import setup_knowledge_base
from translator import (
    detect_language,
    translate_to_english,
    translate_from_english,
    get_language_name
)

from retriever import (
    retrieve_documents,
    build_context
)

from llm import ask_llm

from safety import (
    apply_safety_rules,
    safety_message
)


# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="KrishiMitra AI",
    page_icon="🌾",
    layout="centered"
)


# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("🌾 KrishiMitra AI")
st.subheader("Multilingual Agricultural Assistant")
st.write("தமிழ் • हिंदी • English")


# ------------------------------------------------
# INITIALIZE KNOWLEDGE BASE
# ------------------------------------------------

@st.cache_resource
def initialize():
    return setup_knowledge_base()


collection = initialize()


# ------------------------------------------------
# LANGUAGE SELECTION
# ------------------------------------------------

language = st.selectbox(
    "Language / மொழி / भाषा",
    options=["ta", "hi", "en"],
    format_func=lambda code: {
        "ta": "🇮🇳 தமிழ்",
        "hi": "🇮🇳 हिंदी",
        "en": "🇬🇧 English"
    }[code]
)


# ------------------------------------------------
# CROP SELECTION
# ------------------------------------------------

crop = st.selectbox(
    "Crop / பயிர் / फसल",
    options=["rice", "tomato", "cotton"],
    format_func=lambda value: {
        "rice": "🌾 Rice / நெல் / धान",
        "tomato": "🍅 Tomato / தக்காளி / टमाटर",
        "cotton": "🌱 Cotton / பருத்தி / कपास"
    }[value]
)


# ------------------------------------------------
# QUESTION
# ------------------------------------------------

question = st.text_area(
    "Your Question / உங்கள் கேள்வி / आपका प्रश्न",
    height=150,
    placeholder=(
        "Example:\n"
        "என் நெல் இலைகள் மஞ்சளாகிறது. என்ன செய்ய வேண்டும்?"
    )
)


# ------------------------------------------------
# ASK BUTTON
# ------------------------------------------------

ask_button = st.button(
    "🌾 Get Agricultural Advice",
    use_container_width=True
)


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

if ask_button:

    if not question.strip():
        st.warning("Please enter your agricultural question.")
        st.stop()

    # --------------------------------------------
    # STEP 1: LANGUAGE DETECTION
    # --------------------------------------------

    with st.spinner("Detecting language..."):
        try:
            detected_language = detect_language(question)
        except Exception as error:
            st.error(f"Language detection error: {error}")
            st.stop()

    st.info(f"Detected language: **{get_language_name(detected_language)}**")

    # --------------------------------------------
    # STEP 2: TRANSLATION
    # --------------------------------------------

    with st.spinner("Processing your question..."):
        try:
            english_question = translate_to_english(
                question, detected_language
            )
        except Exception as error:
            st.error(f"Translation error: {error}")
            st.stop()

    # --------------------------------------------
    # STEP 3: RETRIEVAL
    # --------------------------------------------

    with st.spinner("Searching agricultural knowledge..."):
        try:
            retrieved_documents = retrieve_documents(
                english_question, crop=crop, top_k=3
            )
        except Exception as error:
            st.error(f"Knowledge search error: {error}")
            st.stop()

    # --------------------------------------------
    # STEP 4: SAFETY CHECK
    # --------------------------------------------

    safety = apply_safety_rules(english_question, retrieved_documents)

    # --------------------------------------------
    # STEP 5: HANDLE UNSAFE / UNKNOWN
    # --------------------------------------------

    if not safety["safe"]:

        english_answer = safety_message(detected_language)

        st.warning(english_answer)
        st.metric("Confidence", f"{safety['confidence'] * 100:.0f}%")
        st.info("👨‍🔬 Expert verification recommended.")
        st.stop()

    # --------------------------------------------
    # STEP 6: BUILD RAG CONTEXT
    # --------------------------------------------

    context = build_context(retrieved_documents)

    # --------------------------------------------
    # STEP 7: LLM (Groq or Gemini — free)
    # --------------------------------------------

    with st.spinner("Generating agricultural answer..."):
        try:
            english_answer = ask_llm(
                question=english_question,
                context=context,
                crop=crop
            )
        except Exception as error:
            st.error(f"AI error: {error}")
            st.stop()

    # --------------------------------------------
    # STEP 8: TRANSLATE ANSWER
    # --------------------------------------------

    with st.spinner("Preparing answer..."):
        try:
            final_answer = translate_from_english(
                english_answer, detected_language
            )
        except Exception as error:
            st.error(f"Answer translation error: {error}")
            st.stop()

    # --------------------------------------------
    # STEP 9: DISPLAY
    # --------------------------------------------

    st.success("🌾 Agricultural advice")
    st.write(final_answer)

    # --------------------------------------------
    # CONFIDENCE
    # --------------------------------------------

    confidence = safety["confidence"]
    st.metric("Knowledge confidence", f"{confidence * 100:.0f}%")

    # --------------------------------------------
    # SOURCES
    # --------------------------------------------

    with st.expander("📚 Retrieved agricultural sources"):
        for item in retrieved_documents:
            st.write(f"**Crop:** {item['crop']}")
            st.write(f"**Source:** {item['source']}")
            st.write(item["document"])
            st.divider()
