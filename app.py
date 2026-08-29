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
 
st.set_page_config(
    page_title="KrishiMitra AI",
    page_icon="🌾",
    layout="centered"
)
 
st.title("🌾 KrishiMitra AI")
st.subheader("Multilingual Agricultural Assistant")
st.write("தமிழ் • हिंदी • English")
 
@st.cache_resource
def initialize():
    return setup_knowledge_base()
 
collection = initialize()
 
language = st.selectbox(
    "Language / மொழி / भाषा",
    options=["auto", "ta", "hi", "en"],
    format_func=lambda code: {
        "auto": "🌐 Auto-detect",
        "ta": "🇮🇳 தமிழ்",
        "hi": "🇮🇳 हिंदी",
        "en": "🇬🇧 English"
    }[code]
)
 
crop = st.selectbox(
    "Crop / பயிர் / फसल",
    options=["rice", "tomato", "cotton"],
    format_func=lambda value: {
        "rice": "🌾 Rice / நெல் / धान",
        "tomato": "🍅 Tomato / தக்காளி / टमाटर",
        "cotton": "🌱 Cotton / பருத்தி / कपास"
    }[value]
)
 
question = st.text_area(
    "Your Question / உங்கள் கேள்வி / आपका प्रश्न",
    height=150,
    placeholder=(
        "Example:\n"
        "என் நெல் இலைகள் மஞ்சளாகிறது. என்ன செய்ய வேண்டும்?"
    )
)
 
ask_button = st.button(
    "🌾 Get Agricultural Advice",
    use_container_width=True
)
 
if ask_button:
    if not question.strip():
        st.warning("Please enter your agricultural question.")
        st.stop()
 
    if language == "auto":
        with st.spinner("Detecting language..."):
            try:
                detected_language = detect_language(question)
            except Exception as error:
                st.error(f"Language detection error: {error}")
                st.stop()
        st.info(f"Detected language: **{get_language_name(detected_language)}**")
    else:
        detected_language = language
        st.info(f"Language: **{get_language_name(detected_language)}**")
 
    with st.spinner("Processing your question..."):
        try:
            english_question = translate_to_english(
                question, detected_language
            )
        except Exception as error:
            st.error(f"Translation error: {error}")
            st.stop()
 
    with st.spinner("Searching agricultural knowledge..."):
        try:
            retrieved_documents = retrieve_documents(
                english_question, crop=crop, top_k=3
            )
        except Exception as error:
            st.error(f"Knowledge search error: {error}")
            st.stop()
 
    safety = apply_safety_rules(english_question, retrieved_documents)
 
    if not safety["safe"]:
        english_answer = safety_message(detected_language)
        st.warning(english_answer)
        st.metric("Confidence", f"{safety['confidence'] * 100:.0f}%")
        st.info("👨‍🔬 Expert verification recommended.")
        st.stop()
 
    context = build_context(retrieved_documents)
 
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
 
    with st.spinner("Preparing answer..."):
        try:
            final_answer = translate_from_english(
                english_answer, detected_language
            )
        except Exception as error:
            st.error(f"Answer translation error: {error}")
            st.stop()
 
    st.success("🌾 Agricultural advice")
    st.write(final_answer)
 
    confidence = safety["confidence"]
    st.metric("Knowledge confidence", f"{confidence * 100:.0f}%")
 
    with st.expander("📚 Retrieved agricultural sources"):
        for item in retrieved_documents:
            st.write(f"**Crop:** {item['crop']}")
            st.write(f"**Source:** {item['source']}")
            st.write(item["document"])
            st.divider()
 
