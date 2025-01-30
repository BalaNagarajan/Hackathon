import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from constants import openai_key
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Load environment variables
# Set OpenAI API key as environment variable
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit app
# Center the title and add some color
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>Theme Machine</h1>
    """,
    unsafe_allow_html=True
)

# Create a two-column layout
col1, col2 = st.columns(2)

# Upper left content
with col1:
    st.markdown(
        """
        <h2 style='color: #FF5733;'>Dashboard</h2>
        """,
        unsafe_allow_html=True
    )
    st.subheader("Asset Class")
    
    with st.expander("Equity"):
        st.write("Details about Equity")
        
        # Create a DataFrame for the table
        equity_data = {
            "NAME": ["Uranium and nuclear", "Bond proxies", "Infrastructure"],
            "YTD%": ["10%", "4%", "3%"],
            "1YR%" : ["30%", "50%", "5%"],
            "UNIQUE SC" : ["0.9", "0.6", "0.8"]
        }
        df_equity = pd.DataFrame(equity_data)

        # Display the table
        st.table(df_equity)

        ## Adding the pop up screen to display the time series - Starts
         
         # Create a selectbox to choose a theme
        selected_theme = st.selectbox("Select a theme to view time series data:", df_equity["NAME"])
        
        # Display the time series data if "Theme1" is selected
        if selected_theme == "Uranium and nuclear":
            st.write("Time Series Data for Uranium and nuclear")
            
            # Hardcoded time series data
            time_series_data = {
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Stock Price": [100, 105, 110]
            }
            df_time_series = pd.DataFrame(time_series_data)
            
            # Display the time series data
            st.line_chart(df_time_series.set_index("Date"))
    
       
         ## Adding the pop up screen to display the time series - Ends


        
    
    with st.expander("Fixed Income"):
        st.write("Details about Fixed Income")

        # Create a DataFrame for the table
        fixed_data = {
            "NAME": ["HIGH YIELD CONVEXITY", "California Munis"],
            "YTD%": ["12%", "11%"],
            "1YR%": ["11%", "12%"],
            "UNIQUE SC": ["0.4", "0.4"]
        }
        df_fixed = pd.DataFrame(fixed_data)
         # Display the table
        st.table(df_fixed)
    
    with st.expander("Multi Asset"):
        st.write("Details about Multi Asset")
        
          # Create a DataFrame for the table
        multi_asset_data = {
            "NAME": ["LA Recovery"],
            "YTD%": ["-10%"],
            "1YR%": ["1%"],
            "UNIQUE SC": ["0.99"]
        }
        df_multi_asset = pd.DataFrame(multi_asset_data)
         # Display the table
        st.table(df_multi_asset)

    
    with st.expander("Alternatives"):
        st.write("Details about Alternatives")

         # Create a DataFrame for the table
        alternatives_data = {
            "NAME": ["Row1", "Row2", "Row3"],
            "YTD": ["Row1", "Row2", "Row3"],
            "1YR%": ["Row1", "Row2", "Row3"],
            "UNIQUE SC": ["Row1", "Row2", "Row3"]
        }
        df_alternatives = pd.DataFrame(alternatives_data)
         # Display the table
        st.table(df_alternatives)

# Lower left content
with col1:
   
    st.markdown(
        """
        <h2 style='color: #FF5733;'>Research & Insights</h2>
        """,
        unsafe_allow_html=True
    )
     # Display news links as row-by-row line items
    st.write("Latest Market News:")
    news_links = {
        "Bloomberg": "https://www.bloomberg.com",
        "Yahoo Finance": "https://finance.yahoo.com",
        "CNN Markets": "https://www.cnn.com/markets"
    }
    
    for name, url in news_links.items():
        st.markdown(f"[{name}]({url})", unsafe_allow_html=True)


    st.write("- Latest Research")
    st.write("- News Summary")

     # Add a text box to enter the user query and a button to submit the user prompt to OpenAI
    user_query = st.text_input("Enter your query:")
    if st.button("Submit"):
        if user_query:
            # Use LangChain's OpenAI integration to handle the query
            llm = OpenAI(api_key=openai_key)
            response = llm(user_query)
            st.write("Response from OpenAI:")
            st.write(response)

             # Add a clear button to refresh the LLM response
    if st.button("Clear"):
        st.session_state.response = ""

    # Display the response if available
    if "response" in st.session_state and st.session_state.response:
        st.write("Response from OpenAI:")
        st.write(st.session_state.response)


# Right content
with col2:
    st.header("CoPilot [Co-CIO]")
    st.write("This is the right column where you can add your dashboard elements.")