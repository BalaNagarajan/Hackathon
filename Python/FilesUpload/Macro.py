import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

#youtube_transcript_api

def test_youtube_transcript(video_url):
    transcript = YouTubeTranscriptApi.get_transcript(video_url, languages=['en'])
    summary_str = " ".join([item['text'] for item in transcript])
    return summary_str

# Streamlit app
st.title("Macro Insights")

# Text box to enter YouTube video URL
video_url = st.text_input("Enter YouTube video URL:")

# Submit button
if st.button("Submit"):
    if video_url:
        try:
            summary = test_youtube_transcript(video_url)
            st.write("Transcript Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please enter a valid YouTube video URL.")