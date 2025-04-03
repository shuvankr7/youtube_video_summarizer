import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from pytube import YouTube
import requests
import random
import time
from urllib.parse import urlparse
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Dict, Any
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'translated_summary' not in st.session_state:
    st.session_state.translated_summary = None
if 'url' not in st.session_state:
    st.session_state.url = None
if 'language_code' not in st.session_state:
    st.session_state.language_code = None
if 'url_input' not in st.session_state:
    st.session_state.url_input = ""
if 'user_api_key' not in st.session_state:
    st.session_state.user_api_key = ""

# List of supported languages for translation
SUPPORTED_LANGUAGES = [
    {"name": "English", "code": "en"},
    {"name": "Bengali", "code": "bn"},
    {"name": "Hindi", "code": "hi"},
    {"name": "Spanish", "code": "es"},
    {"name": "French", "code": "fr"},
    {"name": "German", "code": "de"},
    {"name": "Italian", "code": "it"},
    {"name": "Portuguese", "code": "pt"},
    {"name": "Russian", "code": "ru"},
    {"name": "Japanese", "code": "ja"},
    {"name": "Korean", "code": "ko"},
    {"name": "Chinese (Simplified)", "code": "zh"},
    {"name": "Arabic", "code": "ar"},
    {"name": "Dutch", "code": "nl"},
    {"name": "Turkish", "code": "tr"},
    {"name": "Polish", "code": "pl"},
    {"name": "Ukrainian", "code": "uk"},
    {"name": "Vietnamese", "code": "vi"},
    {"name": "Thai", "code": "th"},
    {"name": "Indonesian", "code": "id"},
    {"name": "Malay", "code": "ms"},
    {"name": "Swedish", "code": "sv"},
    {"name": "Norwegian", "code": "no"},
    {"name": "Danish", "code": "da"},
    {"name": "Finnish", "code": "fi"},
    {"name": "Greek", "code": "el"},
    {"name": "Hebrew", "code": "he"},
    {"name": "Romanian", "code": "ro"},
    {"name": "Hungarian", "code": "hu"},
    {"name": "Czech", "code": "cs"},
    {"name": "Slovak", "code": "sk"},
    {"name": "Croatian", "code": "hr"},
    {"name": "Serbian", "code": "sr"},
    {"name": "Bulgarian", "code": "bg"},
    {"name": "Slovenian", "code": "sl"},
    {"name": "Estonian", "code": "et"},
    {"name": "Latvian", "code": "lv"},
    {"name": "Lithuanian", "code": "lt"},
    {"name": "Icelandic", "code": "is"},
    {"name": "Maltese", "code": "mt"},
    {"name": "Welsh", "code": "cy"},
    {"name": "Irish", "code": "ga"},
    {"name": "Scottish Gaelic", "code": "gd"},
    {"name": "Manx", "code": "gv"},
    {"name": "Cornish", "code": "kw"},
    {"name": "Breton", "code": "br"},
    {"name": "Basque", "code": "eu"},
    {"name": "Catalan", "code": "ca"},
    {"name": "Galician", "code": "gl"},
    {"name": "Afrikaans", "code": "af"},
    {"name": "Swahili", "code": "sw"},
    {"name": "Zulu", "code": "zu"},
    {"name": "Xhosa", "code": "xh"},
    {"name": "Yoruba", "code": "yo"},
    {"name": "Igbo", "code": "ig"},
    {"name": "Hausa", "code": "ha"},
    {"name": "Somali", "code": "so"},
    {"name": "Amharic", "code": "am"},
    {"name": "Oromo", "code": "om"},
    {"name": "Tigrinya", "code": "ti"},
    {"name": "Kinyarwanda", "code": "rw"},
    {"name": "Kirundi", "code": "rn"},
    {"name": "Malagasy", "code": "mg"},
    {"name": "Sesotho", "code": "st"},
    {"name": "Setswana", "code": "tn"},
    {"name": "Siswati", "code": "ss"},
    {"name": "Tsonga", "code": "ts"},
    {"name": "Venda", "code": "ve"},
    {"name": "Ndebele", "code": "nd"},
    {"name": "Shona", "code": "sn"},
    {"name": "Chichewa", "code": "ny"},
    {"name": "Tswana", "code": "tn"},
    {"name": "Sotho", "code": "st"},
    {"name": "Tamil", "code": "ta"},
    {"name": "Telugu", "code": "te"},
    {"name": "Kannada", "code": "kn"},
    {"name": "Malayalam", "code": "ml"},
    {"name": "Gujarati", "code": "gu"},
    {"name": "Marathi", "code": "mr"},
    {"name": "Punjabi", "code": "pa"},
    {"name": "Urdu", "code": "ur"},
    {"name": "Nepali", "code": "ne"},
    {"name": "Sinhala", "code": "si"},
    {"name": "Burmese", "code": "my"},
    {"name": "Khmer", "code": "km"},
    {"name": "Lao", "code": "lo"},
    {"name": "Mongolian", "code": "mn"},
    {"name": "Tibetan", "code": "bo"},
    {"name": "Uyghur", "code": "ug"},
    {"name": "Kazakh", "code": "kk"},
    {"name": "Kyrgyz", "code": "ky"},
    {"name": "Uzbek", "code": "uz"},
    {"name": "Turkmen", "code": "tk"},
    {"name": "Tajik", "code": "tg"},
    {"name": "Pashto", "code": "ps"},
    {"name": "Dari", "code": "prs"},
    {"name": "Kurdish", "code": "ku"},
    {"name": "Persian", "code": "fa"},
    {"name": "Sindhi", "code": "sd"},
    {"name": "Balochi", "code": "bal"},
    {"name": "Kashmiri", "code": "ks"},
    {"name": "Dogri", "code": "doi"},
    {"name": "Konkani", "code": "kok"},
    {"name": "Manipuri", "code": "mni"},
    {"name": "Bodo", "code": "brx"},
    {"name": "Sanskrit", "code": "sa"},
    {"name": "Maithili", "code": "mai"},
    {"name": "Santali", "code": "sat"},
    {"name": "Nepali", "code": "ne"},
    {"name": "Sikkimese", "code": "sip"},
    {"name": "Ladakhi", "code": "lbj"},
    {"name": "Tulu", "code": "tcy"},
    {"name": "Kodava", "code": "kfa"},
    {"name": "Toda", "code": "tcx"},
    {"name": "Badaga", "code": "bfq"},
    {"name": "Kurumba", "code": "kfi"},
    {"name": "Irula", "code": "iru"},
    {"name": "Paniya", "code": "pcg"},
    {"name": "Mullu Kurumba", "code": "kfi"},
    {"name": "Betta Kurumba", "code": "kfi"},
    {"name": "Mala Malasar", "code": "ymr"},
    {"name": "Mala Arayan", "code": "ymr"},
    {"name": "Mannan", "code": "mjv"},
    {"name": "Muthuvan", "code": "muv"},
    {"name": "Hill Pandaram", "code": "pci"},
    {"name": "Malapandaram", "code": "mjp"},
    {"name": "Urali", "code": "url"},
    {"name": "Mannan", "code": "mjv"},
    {"name": "Muthuvan", "code": "muv"},
    {"name": "Hill Pandaram", "code": "pci"},
    {"name": "Malapandaram", "code": "mjp"},
    {"name": "Urali", "code": "url"},
    {"name": "Mannan", "code": "mjv"},
    {"name": "Muthuvan", "code": "muv"},
    {"name": "Hill Pandaram", "code": "pci"},
    {"name": "Malapandaram", "code": "mjp"},
    {"name": "Urali", "code": "url"}
]

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)',
        r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]+)',
        r'(?:youtube\.com\/v\/)([a-zA-Z0-9_-]+)',
        r'(?:youtube\.com\/e\/)([a-zA-Z0-9_-]+)',
        r'(?:youtube\.com\/user\/[^\/]+\/)([a-zA-Z0-9_-]+)',
        r'(?:youtube\.com\/[^\/]+\/live\/)([a-zA-Z0-9_-]+)',
        r'(?:youtube\.com\/[^\/]+\/)([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info(url: str) -> Optional[Dict[str, Any]]:
    """
    Get video information using YouTube Transcript API
    """
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return None

        # Get video thumbnail
        thumbnail_url = f"http://img.youtube.com/vi/{video_id}/0.jpg"
        
        # Get transcript to extract some basic info
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if transcript:
                return {
                    'title': f"Video ID: {video_id}",
                    'thumbnail': thumbnail_url,
                    'has_transcript': True
                }
        except:
            return {
                'title': f"Video ID: {video_id}",
                'thumbnail': thumbnail_url,
                'has_transcript': False
            }

    except Exception as e:
        st.error(f"Error getting video info: {str(e)}")
        return None

def get_video_transcript(url: str) -> Optional[str]:
    """
    Get video transcript using YouTube Transcript API
    """
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return None

        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Show available languages
        st.success("Available languages for this video:")
        for transcript in transcript_list:
            st.write(f"- {transcript.language} ({'Auto-generated' if transcript.is_generated else 'Manually created'})")
        
        # Try to get English transcript first
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            st.success("Using English transcript")
        except:
            # If English not available, try Hindi
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
                st.warning("English transcript not available. Using Hindi transcript instead.")
            except:
                # If neither English nor Hindi available, get the first available language
                first_lang = transcript_list[0].language_code
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[first_lang])
                st.warning(f"Using {transcript_list[0].language} transcript")
        
        # Format transcript
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)
        
        return transcript_text

    except Exception as e:
        st.error(f"Error getting transcript: {str(e)}")
        return None

def get_available_languages(video_id):
    """Get available caption languages using YouTube Transcript API"""
    try:
        # First check if the video exists and has captions enabled
        try:
            languages = YouTubeTranscriptApi.list_transcripts(video_id)
        except Exception as e:
            if "Subtitles are disabled" in str(e):
                st.error("""
                ❌ This video has subtitles disabled by the creator. 
                
                To use this tool, please try a different video that has:
                1. Manual captions added by the creator, or
                2. Auto-generated captions enabled
                
                You can check if a video has captions by:
                1. Opening the video on YouTube
                2. Clicking the "CC" (Closed Captions) button
                3. If no captions are available, try another video
                """)
                return None
            else:
                st.error(f"Error getting available languages: {str(e)}")
                return None
        
        available_languages = []
        for transcript in languages:
            available_languages.append({
                'code': transcript.language_code,
                'name': transcript.language,
                'is_generated': transcript.is_generated
            })

        if not available_languages:
            st.error("""
            ❌ No captions available for this video.
            
            This could be because:
            1. The video creator has disabled captions
            2. The video is too new and auto-captions haven't been generated yet
            3. The video is private or unlisted
            
            Please try a different video that has captions enabled.
            """)
            return None

        # Show available languages
        st.success("Available languages for this video:")
        for lang in available_languages:
            st.write(f"- {lang['name']} ({'Auto-generated' if lang['is_generated'] else 'Manually created'})")

        return available_languages

    except Exception as e:
        st.error(f"Error getting available languages: {str(e)}")
        return None

def get_video_duration(url):
    """Get video duration using pytube"""
    try:
        yt = YouTube(url)
        duration_seconds = yt.length
        return duration_seconds
    except Exception as e:
        # Silently continue if duration check fails
        return None

def get_api_key():
    """Get API key with priority to user provided key"""
    if st.session_state.user_api_key:
        return st.session_state.user_api_key
    return os.getenv("GROQ_API_KEY")

def summarize_video(url, language_code='en'):
    """Summarize YouTube video content using Groq"""
    # Get GROQ_API_KEY with priority to user provided key
    GROQ_API_KEY = get_api_key()
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Please provide your API key in the settings above.")
        return None

    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL")
        return None

    # Get available languages
    available_languages = get_available_languages(video_id)
    if not available_languages:
        st.error("Could not retrieve available languages for this video")
        return None

    # If requested language is not available, use the first available language
    if not any(lang['code'] == language_code for lang in available_languages):
        language_code = available_languages[0]['code']
        st.info(f"Requested language not available. Using {available_languages[0]['name']} instead.")

    # Get video transcript
    transcript = get_video_transcript(url)
    if not transcript:
        return None

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )

    # Split transcript into chunks
    docs = text_splitter.create_documents([transcript])

    # Initialize Groq with llama-3.1-8b-instant model
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=2048
    )

    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""Please provide a comprehensive summary of the following text. 
        Focus on the main points and key details while maintaining the original context and meaning:

        {text}

        SUMMARY:"""
    )

    # Initialize chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate summary
    try:
        # Process each chunk and combine summaries
        summaries = []
        for doc in docs:
            result = chain.invoke({"text": doc.page_content})
            summaries.append(result['text'])

        # Combine all summaries
        combined_text = "\n\n".join(summaries)
        
        # Create final summary
        final_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Please provide a final, concise summary combining all these points:

            {text}

            FINAL SUMMARY:"""
        )
        
        final_chain = LLMChain(llm=llm, prompt=final_prompt)
        final_result = final_chain.invoke({"text": combined_text})
        
        return final_result['text']
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def translate_summary(summary, target_language='en'):
    """Translate summary to target language using Groq"""
    # Get GROQ_API_KEY with priority to user provided key
    GROQ_API_KEY = get_api_key()
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Please provide your API key in the settings above.")
        return None

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=2048
    )

    translate_prompt = PromptTemplate(
        input_variables=["text", "target_language"],
        template="""Please translate the following text to {target_language}. 
        Maintain the same format and structure while ensuring accurate translation:

        {text}

        TRANSLATION:"""
    )

    chain = LLMChain(llm=llm, prompt=translate_prompt)
    
    try:
        result = chain.invoke({
            "text": summary,
            "target_language": target_language
        })
        return result['text']
    except Exception as e:
        st.error(f"Error translating summary: {str(e)}")
        return None

def summarize_content(content: str) -> str:
    """
    Summarize the content using Groq
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes content concisely."),
            ("user", "Please summarize the following content in bullet points:\n\n{content}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"content": content})
    except Exception as e:
        st.error(f"Error summarizing content: {str(e)}")
        return ""

def generate_summary(transcript: str) -> str:
    """Generate summary using Google's Gemini Pro model"""
    try:
        prompt = """You are a YouTube video summarizer. You will be taking the transcript text
        and summarizing the entire video and providing the important summary in points
        within 250 words. Please provide the summary of the text given here: """

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript)
        return response.text

    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return ""

def main():
    st.set_page_config(
        page_title="YouTube Video Summarizer",
        page_icon="🎥",
        layout="centered"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #FF0000;
            color: white;
        }
        .stButton>button:hover {
            background-color: #CC0000;
            color: white;
        }
        .success-text {
            color: #28a745;
            font-weight: bold;
        }
        .warning-text {
            color: #ffc107;
            font-weight: bold;
        }
        .error-text {
            color: #dc3545;
            font-weight: bold;
        }
        .title-text {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 1em;
            color: #1E1E1E;
        }
        .subtitle-text {
            text-align: center;
            font-size: 1.2em;
            color: #666666;
            margin-bottom: 2em;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and Description with custom styling
    st.markdown('<p class="title-text">🎥 YouTube Video Summarizer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Get quick, accurate summaries of YouTube videos in multiple languages</p>', unsafe_allow_html=True)

    # API Settings in expander (full width)
    with st.expander("⚙️ API Settings"):
        st.info("If you face any API limitations, you can provide your own Groq API key. Get one at https://console.groq.com . They provide free API key upto a limit, please check their website")
        user_api_key = st.text_input(
            "Groq API Key (Optional)", 
            value=st.session_state.user_api_key,
            type="password",
            help="Your API key will be used instead of the default key if provided"
        )
        if user_api_key != st.session_state.user_api_key:
            st.session_state.user_api_key = user_api_key
            if user_api_key:
                st.success("✅ API key updated!")

    # Main content in a container for better organization
    with st.container():
        # URL input section with improved styling
        st.markdown("### Enter Video URL")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            url_input = st.text_input(
                "",  # Remove label as we have the header above
                placeholder="https://youtube.com/watch?v=...",
                value=st.session_state.url_input,
                key="url_input_field"
            )
        
        with col2:
            st.write("")  # For vertical alignment
            if st.button("Fetch Transcripts", key="url_enter_button", help="Click to fetch available transcripts"):
                st.session_state.url_input = url_input
                st.session_state.url = url_input
                st.session_state.summary = None
                st.session_state.translated_summary = None
                st.session_state.language_code = None

        # Warning about video length with custom styling
        st.markdown("""
            <div style='background-color: #fff3cd; padding: 1em; border-radius: 5px; margin: 1em 0;'>
                ⚠️ <span style='color: #856404;'>This system is optimized for videos under 10 minutes. 
                Longer videos may result in incomplete summaries or processing delays.</span>
            </div>
        """, unsafe_allow_html=True)
        
        url = st.session_state.url_input if st.session_state.url_input else None
        
        if url:
            # Get video ID and show processing status
            video_id = extract_video_id(url)
            if not video_id:
                st.error("❌ Invalid YouTube URL")
                return

            # Check video duration
            duration = get_video_duration(url)
            if duration and duration > 600:
                st.error("❌ This video is longer than 10 minutes. For best results, please use a shorter video.")
                return

            # Get available languages with loading animation
            with st.spinner("🔍 Fetching available languages..."):
                available_languages = get_available_languages(video_id)
                if not available_languages:
                    st.error("❌ Could not retrieve available languages for this video")
                    return

            # Language selection with improved UI
            st.markdown("### Select Source Language")
            language_options = [f"{lang['name']} ({lang['code']}){' [Auto-generated]' if lang['is_generated'] else ''}" 
                              for lang in available_languages]
            selected_language = st.selectbox(
                "",  # Remove label as we have the header above
                language_options
            )
            language_code = available_languages[language_options.index(selected_language)]['code']

            # Generate summary section
            if st.button("Generate Summary", help="Click to generate video summary"):
                with st.spinner("🔄 Generating summary..."):
                    st.session_state.summary = summarize_video(url, language_code)
                    st.session_state.language_code = language_code
                    st.session_state.translated_summary = None

            # Display summary and translation options
            if st.session_state.summary:
                st.markdown("### 📝 Video Summary")
                st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 1.5em; border-radius: 5px; margin: 1em 0;'>
                        {st.session_state.summary}
                    </div>
                """, unsafe_allow_html=True)
                
                # Translation section with improved UI
                st.markdown("### 🌐 Translation Options")
                target_language = st.selectbox(
                    "Select Target Language (Default Best option is English)",
                    options=[lang["name"] for lang in SUPPORTED_LANGUAGES],
                    index=0
                )
                
                if st.button("Translate", help="Click to translate the summary"):
                    with st.spinner(f"🔄 Translating to {target_language}..."):
                        st.session_state.translated_summary = translate_summary(
                            st.session_state.summary, 
                            target_language
                        )

            # Display translation with styling
            if st.session_state.translated_summary:
                st.markdown(f"### 🌍 {target_language} Translation")
                st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 1.5em; border-radius: 5px; margin: 1em 0;'>
                        {st.session_state.translated_summary}
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
