import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import plotly.graph_objs as py
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','']) # remove it if you need punctuation 
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# main functions to get the comfort food sorted from the day as per the reason

def searchComfy(mood):
    lemmatizer = WordNetLemmatizer()
    foodcount = {}
    for i in range(124):
        temp = [temps.strip().replace('.','').replace(',','').lower() for temps in str(fc["comfort_food_reasons"][i]).split(' ') if temps.strip() not in stop ]
        if mood in temp:
            foodtemp = [lemmatizer.lemmatize(temps.strip().replace('.','').replace(',','').lower()) for temps in str(fc["comfort_food"][i]).split(',') if temps.strip() not in stop ]
            for a in foodtemp:
                if a not in foodcount.keys():
                    foodcount[a] = 1 
                else:
                    foodcount[a] += 1
    sorted_food = []
    #print(sorted([(value,key) for (key,value) in foodcount.items()])) : for test
    sorted_food = sorted(foodcount, key=foodcount.get, reverse=True)
    return sorted_food


def findmycomfortfood(mood):
    topn = []
    topn = searchComfy(mood) #function create dictionary only for particular mood
    return topn

#main function to plot definition
#coffee: 1-Creamy Frappuccino & 2-Espresso; breakfast: 1-cereal option & 2 – donut option; 
#soup: 1 – veggie soup and 2 – creamy soup; drink: 1 – orange juice and 2 – soda 
from itertools import cycle
cycol = cycle('kbgc')

st.header('EDA on Food Choices and prefs of Students')
df ,ri,fc,sl,CountryInfo= None,None,None,None,None

@st.cache
def load_data():
    global df,ri,fc,sl
    df = pd.read_csv("input/zomato-restaurants-data/zomato.csv",encoding='latin-1') #Main restaurants data
    ri = pd.read_json("input/recipe-ingredients-dataset/train.json")                #Cuisines recipe ingredients
    fc = pd.read_csv("input/food-choices/food_coded.csv")                           #Food Choice data of college students
    sl = pd.read_csv("input/store-locations/directory.csv")                         #starbucks data

    CountryInfo = pd.read_excel('input/zomato-restaurants-data/Country-Code.xlsx')  
    return df,ri,fc,sl,CountryInfo
with st.spinner("loading datasets"):
    df,ri,fc,sl,CountryInfo = load_data()                  
    st.success('data loaded successfully')

with st.beta_expander('Contents in the project'):
    st.image('restaurants.jpg',use_column_width=True)
    st.markdown('''
    1. What are the some Cities around the globe with Maximum Restaurants?
    2. What are the Top10 Popular Ratings as per the Counts?
    3. Word Cloud Visualization: for the Cuisines (as per its frequencies)
    4. What are the Top20 Cuisines?
    5. Where are the Restaurants hiding & what's their Ratings?
    6. Most Common Ingredients used in Cuisines Worldwide!
    7. Top Comfort Foods
    8. Popular Comfort Food choices when you are Stressed out!
    9. Is there a relation between Average GPA and Food Choices?
    10. How Expensive are the Restaurants?
    11. Eating Out Frequency as per Marital Status
    12. Where are the Starbucks Stores Located ?
    13. What are the Top10 Indian Cities with Maximum Restaurants?
    14. Word Cloud: for the Restaurant Name (What are some popular Indian Restaurant Names)?
    15. How are the Rating types ('Excellent', 'Poor' etc) represented as per the Average Ratings (in India)?
    ''')

with st.beta_expander('1. What are the some Cities around the globe with Maximum Restaurants?'):
    wordcloud = (WordCloud( max_words=100,width=700, height=300,max_font_size=60, min_font_size=10, relative_scaling=0.5,background_color='white').generate_from_frequencies(df['City'].value_counts()))
    fig,ax = plt.subplots()
    plt.imshow(wordcloud,interpolation="gaussian")
    plt.axis('off')
    st.pyplot(fig)

with st.beta_expander('2. What are the Top10 Popular Ratings as per the Counts?'):
    TopRatings=df.groupby(['Aggregate rating'],as_index = False)['Restaurant ID'].count()
    TopRatings.columns = ['Rating','Counts']
    Top = TopRatings.sort_values(by='Counts',ascending=False).head(11).reset_index(drop=True)
    Top = Top.iloc[1:,:]
    fig,ax = plt.subplots(figsize=(10,6))
    sns.barplot(Top['Rating'],Top['Counts'])
    plt.title("Top 10 Ratings with count")
    st.pyplot(fig)

with st.beta_expander('3. Word Cloud Visualization: for the Cuisines (as per its frequencies)'):
    wordcloud = (WordCloud(width=500, height=300, relative_scaling=0.5, stopwords=stopwords, background_color='white').generate_from_frequencies(df['Cuisines'].value_counts()))
    fig,ax = plt.subplots(figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis('off')
    st.pyplot(fig)

with st.beta_expander('4. What are the Top20 Cuisines?'):
    cuisines_data = df.groupby(['Cuisines'],as_index = False)['Restaurant ID'].count()
    #now time to get most popular cusinies on board, we know North Indian gonna hit the list
    cuisines_data.columns = ['Popular Cusinies','Number of Restaurants']
    cuisines_data.reindex(axis="index")
    cuisines_data.sort_values(by='Number of Restaurants',ascending=False,inplace=True)
    cuisines_data.head(20).reset_index(drop=True)
    st.dataframe(cuisines_data)

with st.beta_expander('5. Where are the Restaurants hiding & what\'s their Ratings?'):
    data1 = [dict(
    type='scattergeo',
    lon = df['Longitude'],
    lat = df['Latitude'],
    text = df['Restaurant Name'],
    mode = 'markers',
    marker = dict(
    cmin = 0,
    color = df['Aggregate rating'],
    cmax = df['Aggregate rating'].max(),
    colorbar=dict(
                title="Ratings"
            )
    )
    
    )]
    layout = dict(
        title = 'Where are the Resturants',
        hovermode='closest',
        geo = dict(showframe=False, countrywidth=1, showcountries=True,
                showcoastlines=True, showocean = True, )
    )
    fig = py.Figure(data=data1, layout=layout)
    st.plotly_chart(fig)


with st.beta_expander('6. Most Common Ingredients used in Cuisines Worldwide!'):
    #Data Prep for below visual
    a = []
    for i in ri["ingredients"]:
        for j in i:
            a.append(j)
    commonIng = pd.DataFrame()
    commonIng["common"] = pd.Series(a)

    #using recipe-ingredients-dataset
    fig,ax = plt.subplots()
    wordcloud = (WordCloud( relative_scaling=0.5, stopwords=stopwords, max_words=100,max_font_size=60, min_font_size=10,background_color='black').generate_from_frequencies(commonIng["common"].value_counts()))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis('off')
    st.pyplot(fig)

with st.beta_expander('7. Top Comfort Foods'):
    #Data prep for below visual
    a = []
    for i in fc["comfort_food"]:
        i = str(i)
        for j in list(i.split(',')):
            a.append(j)
    comfy = pd.DataFrame()
    comfy["food"] = pd.Series(a)
    #what's my comfort food -> using Food_choices dataset
    fig,ax = plt.subplots()
    wordcloud = (WordCloud(relative_scaling=0.5, stopwords=stopwords,max_words=1000,background_color='white').generate_from_frequencies(comfy["food"].value_counts()))
    plt.imshow(wordcloud,interpolation="gaussian")
    plt.axis('off')
    st.pyplot(fig)

with st.beta_expander('8. Popular Comfort Food choices when you are Stressed out!'):
    topn = findmycomfortfood('stress')
    st.info("3 Popular Comfort Foods in %s are:"%('stress'))
    st.success(topn[0])
    st.success(topn[1])
    st.success(topn[2])

with st.beta_expander('9. Popular Cuisine choices when you are Stressed out!'):
    #GPA Data Prep
    dfc = fc
    dfc = dfc.drop(dfc.index[15])
    dfc = dfc.drop(dfc.index[60])
    dfc = dfc.drop(dfc.index[100])
    dfc = dfc.drop(dfc.index[101])
    dfc.reset_index(inplace=True)
    dfc["GPA"][71]= str(dfc["GPA"][71]).replace(' bitch','')
    dfc["GPA"] = pd.to_numeric(dfc["GPA"])

    def barGPA(category,label1, label2):
        barGPA = {} 
        barGPA[label1] = dfc[dfc[category]==1]["GPA"].mean() 
        barGPA[label2] = dfc[dfc[category]==2]["GPA"].mean()
        D = barGPA
        fig,ax = plt.subplots(figsize=(5,4)) 
        plt.bar(range(len(D)), list(D.values()), align='center',color=next(cycol))
        plt.xticks(range(len(D)), list(D.keys()))
        plt.ylim(ymin=2) 
        plt.yticks(np.arange(2, 3.6, 0.1))
        plt.ylabel("GPA")
        plt.title(category.upper()+" vs GPA")
        return fig

    st.pyplot(barGPA('coffee','Frappuccino','Espresso'))
    st.pyplot(barGPA('drink','Orange Juice','Soda'))
    st.pyplot(barGPA('breakfast','Cereal','Donut'))
    st.pyplot(barGPA('soup','Veggie Soup','Creamy Soup'))

with st.beta_expander('10. How Expensive are the Restaurants?'):
    dfcopy = df.copy()
    dfcopy["Price range"] = dfcopy["Price range"].astype(str)
    dfcopy["Price range"] = dfcopy["Price range"].str.replace('1', '₹').replace('2', '₹₹').replace('3', '₹₹₹').replace('4', '₹₹₹₹')
    fig,ax = plt.subplots()
    plt.title("Distribution of Price Ranges")
    #plt.bar(x=[1,2,3,4],height=[2,4,8,16],hatch='x',edgecolor=['black']*6)
    sns.countplot(dfcopy["Price range"],order=['₹','₹₹','₹₹₹','₹₹₹₹'],hatch='x')
    plt.xlabel("Price Range (low to high)",color='green')
    plt.ylabel("No. of Restaurants",color='green')
    st.pyplot(fig)



with st.beta_expander('11. Eating Out Frequency as per Marital Status'):
    
    dfc = fc
    dfc = dfc.dropna(subset=['marital_status','eating_out'])
    dfc['marital_status']=dfc['marital_status'].astype(str)
    dfc['marital_status'] = dfc['marital_status'].str.replace('1.0','Single')\
        .replace('2.0','In a relationship').replace('4.0','Married')  #since only 3 kind of Marital status available in data
    AvgEatOut = dfc.groupby('marital_status')['eating_out'].mean()

    #plotting
    fig,ax = plt.subplots(figsize=(15,9))
    plt.bar(x=AvgEatOut.index, height =AvgEatOut.values , color='white',edgecolor=['red']*3,hatch='*')
    plt.xlabel('Marital Status of Students (as per Data)',color='green')
    plt.ylabel('Freq. of Eating Out (Average)',color='green')
    plt.title("Eating Out Frequency as per Marital Status of Students",color='grey')
    st.pyplot(fig)

with st.beta_expander('12. What is the frequency of Eating Out?'):
    data1 = [dict(
    type='scattergeo',
    lon = sl['Longitude'],
    lat = sl['Latitude'],
    text = sl['Store Name'],
    mode = 'markers',
    marker = dict(
    color = '#6f4e37',
    )          
    
    )]
    layout = dict(
        title = 'Where is that Starbucks?',titlefont=dict(color='green'), 
        hovermode='closest',
        geo = dict(showframe=False, countrywidth=1, showcountries=True,showocean=True,showland=True,countrycolor='green', 
                showcoastlines=True, projection=dict(type='natural earth')),
        
    )
    fig = py.Figure(data=data1, layout=layout)
    st.plotly_chart(fig)

df_india = df[df['Country Code']==1]

TopCities = df_india.groupby(['City'],as_index = False)['Restaurant ID'].count()
TopCities.columns = ['Cities','Number of Restaurants']
Top10 = TopCities.sort_values(by='Number of Restaurants',ascending=False).head(10)

with st.beta_expander('13. Top 10 Cities with the most number of Restaurants'):
    fig,ax = plt.subplots(figsize=(10,6))
    plt.bar(Top10['Cities'],Top10['Number of Restaurants'], color ='green')
    plt.xlabel('Cities')
    plt.ylabel('Number of Restaurants')
    plt.title('Indian Cities with Maximum Restaurants', fontweight="bold")
    st.pyplot(fig)

with st.beta_expander('14. Word Cloud: for the Restaurant Name (What are some popular Indian Restaurant Names)?'):
    fig,ax = plt.subplots(figsize=(10,6))
    #lets see which restaurant got the most shout-out 
    #Grey color function
    def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)

    wc = (WordCloud(relative_scaling=0.5, stopwords=stopwords).generate_from_frequencies(df_india['Restaurant Name'].value_counts()))
    plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),interpolation="bilinear")
    plt.axis('off')
    st.pyplot(fig)

with st.beta_expander('15.How are the Rating types ("Excellent", "Poor", etc) represented as per the Average Ratings (in India)?'):
    fig,ax = plt.subplots(figsize=(10,6))
    df["Rating color"].value_counts()
    Color_represents = df_india.groupby(['Rating color'],as_index = False)['Aggregate rating'].mean()
    Color_represents.columns = ['Rating Color','Average Rating']
    Color_represents =Color_represents.sort_values(by='Average Rating',ascending=False)
    Color_represents = Color_represents[0:5]
    Color_represents['Ratings']  = ['1.Excellent','2.Very Good','3.Good','4.Okay','5.Poor']
    #Color_represents
    a = ['#006400','green','yellow','Orange','Red']
    e = ['blue']*5
    plt.barh(Color_represents['Ratings'],Color_represents['Average Rating'], align='center', color =a, edgecolor = e, linewidth = 3)
    plt.gca().invert_yaxis()
    plt.xlabel('Average Rating')
    plt.ylabel('Rating Types')
    plt.title('Rating Color and the Average Rating they represent', fontweight="bold")
    plt.show()
    st.pyplot(fig)
