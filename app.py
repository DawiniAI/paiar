import streamlit as st 
import pandas as pd
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from PIL import Image
logo = Image.open('./media/logo.png')


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Paiar| Dawini.ai Product', initial_sidebar_state = 'auto')


from automatedPloting import createPlotForStreamlit

st.subheader('Paiar will help you to automated data analysis task , Lets Start ..')
col1, col2, col3 = st.columns(3)
with col2:    
    st.image(logo, caption='Dawini.ai | Paiar Product',width=100)

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

if uploaded_file is not None:




    st.subheader("Meta Data Breif:")
    with st.spinner('1. [META DATA] of Report is Creating Right now ..'):

        df = pd.read_csv(uploaded_file,encoding = "ISO-8859-1")
        st.write(df.head(3))


        agent = create_pandas_dataframe_agent(
            OpenAI(temperature=0),
            df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        


        ## How Many Rows Are There 
        output = agent.run("how many rows are there?")
        st.write(f"**{output}**")
        output1 = agent.run("how many columns are there and give me description and brief for eachone from your view")
        st.write(f"**{output1}**")
        output2 = agent.run("what type of the dataset is this dataset , i mean by type what domain in business sector")
        st.write(f"**{output2}**")

    
        st.subheader("Knowledge Base :")
        with st.spinner('2.[Knowledge Base] of Report is Creating Right now ..'):

                ## How Many Country Serve , What states Are There 
                output = agent.run("how many country this company serve and the name of the countries")
                st.write(f'**{output}**')
                output1 = agent.run("is there any state the company serve inside the serving country and give me there names")
                st.write(f"**{output1}**")


                st.subheader("Sales Analysis Report :")
                with st.spinner('3.[Sales Report] of Report is Creating Right now ..'):
                    ## How Many orders Are There 
                    fig_1 = createPlotForStreamlit(
                            df,"plot a pie chart for a report for orders count inside countries and city and state inside this dataset , make it for me with diffrent colors for each "
                    )
                    st.pyplot(
                            fig_1,use_container_width=True
                    )
                    output= agent.run("if is possible to make a breif report for orders inside countries and city and state inside this dataset , make it for me")
                    st.write(f"**{output}**")


                    ## How Much sales by orders Are There 
                    fig_2 = createPlotForStreamlit(
                            df,"plot a chart for sales inside  cites and state inside this dataset , make it for me with diffrent colors for each "
                    )
                    st.pyplot(
                            fig_2,use_container_width=True
                    )
                    output1 = agent.run("if is possible to make a breif report for orders sales sum inside cities and state inside this dataset ")
                    st.write(f"**{output1}**")


                    ## How Much sales by orders (monthByMonth)Are There 
                    fig_3 = createPlotForStreamlit(
                            df,"make a plot bar for orders sales by month inside countries and city and state inside this dataset"
                    )
                    st.pyplot(
                            fig_3,use_container_width=True
                    )
                    output2 = agent.run(" make a brief report for orders sales by month inside countries and city and state inside this dataset")
                    st.write(f" **{output2}**")


                    ## how much is the mean sales by orders Are There 
                    fig_4 = createPlotForStreamlit(
                            df,"make a plot for mean sales by city and state inside this dataset"
                    )
                    st.pyplot(
                            fig_4,use_container_width=True
                    )
                    output3 = agent.run("make breif report mean sales for each city per day ")
                    st.write(f"**{output3}**")

                    
                    st.subheader(" Business Intelligence Notice :")
                    with st.spinner('4.[Business Intelligence Notice] of Report is Creating Right now ..'):

                            ## How Many Rows Are There 
                            output = agent.run("make a brief report for best months to sell promocodes to increase the sales , type only breif text for  the monthes name with low orders")
                            st.write(f"**{output}**")
                
            

                    





        
    




    





