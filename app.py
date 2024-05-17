import streamlit as st
import back
import os
import json  # Don't forget to import json
from tempfile import NamedTemporaryFile
import pandas as pd

def main():
    st.set_page_config(page_title="Pandata - AI Video Tagger")
    st.image("logo.png", width=150)
    st.title('AI Video Tagger')
    video_file = st.file_uploader("Video hochladen", type=["mp4", "avi", "mov"])
    tags_to_return = st.slider("Wieviele Tags sollen generiert werden?", min_value=5, max_value=50, value=20)

    if st.button('Tags generieren') and video_file is not None:

        with st.spinner('Videoanalyse...'):
            st.write("Je nach Länge das Videos kann das 1 -2 Minuten dauern")
            # Save the uploaded file to a temporary file
            with NamedTemporaryFile(delete=False) as temp_video_file:
                temp_video_file.write(video_file.read())
                temp_video_path = temp_video_file.name

            tags_json = back.main(temp_video_path, tags_to_return)
            tags = json.loads(tags_json)

            # Create a DataFrame to display as a table
            tags_df = pd.DataFrame(tags["tags"])

            st.subheader("Generierte Tags:")
            st.dataframe(tags_df,hide_index=1)

if __name__ == "__main__":
    main()
