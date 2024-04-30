import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

file_path=r"S:\satya\python\hpp\bengaluru_house_prices.csv"

df=pd.read_csv(file_path)

df.head()
df.shape

#counting the [areatype] based on types
df.groupby('area_type')['area_type'].agg('count')

#drop unimportant columns for our calculation
df1=df.drop(['area_type',"availability","society",'balcony'],axis='columns')

df1.isnull().sum()

#removing numm values
df2=df1.dropna()

df2.isnull().sum()

#[size] features
df2.shape

df2['size'].unique()

#applying function using lambda
df2['bhk']=df2['size'].apply(lambda x: int(x.split(' ')[0]))

df2['bhk'].unique()

df2[df2.bhk>20]

df2.total_sqft.unique()

#method to overcome x-y type values in [total_sqft]
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

#non uniformity in values
df2[~df2['total_sqft'].apply(is_float)].head(10)

#function to convert the x-y values by having average=>x+y/2
def convert_sqft_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

#new df again

df3=df2.copy()
df3.total_sqft=df3['total_sqft'].apply(convert_sqft_to_num)

df3.head()

#check required row
df3.loc[30]

df4=df3.copy()

#calculating price per sqft
df4['price_per_sqft']=df4['price']*100000/df4['total_sqft']

df4.head()

#[location](categorical feature)

len(df4['location'].unique())

#many unique values.. dimensionality curse... 

df4.location=df4.location.apply(lambda x: x.strip())

location_stats=df4.groupby('location')['location'].agg('count').sort_values(ascending=False)

len(location_stats[location_stats <=10])

#we will catregorize them based on range..(linke below 10 one category and rest another)
location_stats_less_than_10=location_stats[location_stats<=10]

len(df4.location.unique())

#separating to categories, if<10 them make it 'other' else put it as same name
df4.location=df4.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)
len(df4.location.unique())

df4.head(10)

#outliers detection and removal
#set a value by asing your manager and set a threshold{a value or range manually given by
#us} for [size]
df4[df4.total_sqft/df4.bhk<300].head()

#from above line of code, we get irregularities as output

df4.shape

df5=df4.copy()

#removing outliers

#taking those value that are > 300 
df5=df5[~(df5.total_sqft/df5.bhk<300)]

df5.shape

df5.price_per_sqft.describe()

#we get max some 176470..., exterem case, 
#may be true but can be removed

#filtering out based on beyond one standard deviation 
#removing outliers

def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df], ignore_index=True)
    
    return df_out

df6=remove_pps_outliers(df5)
df6.shape

#use commonsense and compare property price for 2 bed and 3 bed
#visualize
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.bhk==2)]
    bhk3=df[(df.location==location)&(df.bhk==3)]
    plt.scatter(bhk2.total_sqft, bhk2.price,color="blue",label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+',color="green",label='3 BHK', s=50)
    plt.xlabel("TOTAL SQFT AREA")
    plt.ylabel("PRICE PER SQFT")
    plt.title(location)
    plt.legend()
    

plot_scatter_chart(df6,"Hebbal")
# from graph, we'll know that price2bhk>price3bhk(outlier)
#well create a fxn and based on business logic, we'll remove outliers
#val of sqft of x no of bedroom must be > val of mean of x-1 no of bedroom
def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats=bhk_stats.get(bhk-1)
                if stats and stats['count']>5:
                    exclude_indices=np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


df7=remove_bhk_outliers(df6)

df7.shape

plot_scatter_chart(df7,"Hebbal")

#histogram to see availability i.e, count per sqft area
plt.hist(df7.price_per_sqft,rwidth=0.8)
plt.xlabel('Price per sqft')
plt.ylabel('count')

#relation of bathrooms and bedrooms( discuss with business manager)    
#no of bathrooms >no of bed rooms+2=> it's a outlier   
df7.bath.unique()

df7[df7.bath>10]

plt.hist(df7.bath,rwidth=0.8)
plt.xlabel('No of bathrooms')
plt.ylabel('count')

df7[df7.bath>df7.bhk+2]
#remove the above code outputs 
df8=df7[df7.bath<df7.bhk+2]
df8.shape

#drop price per sqft and size 
df9=df8.drop(['size','price_per_sqft'], axis='columns')
df9.head(3)

#create dummies from [location]=> one hot encoding
dummies=pd.get_dummies(df9.location, dtype=int) 
dummies.head(2)

#dump last column, to avoid dummy variable trap

df10=pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
df10.head(2)
df11=df10.drop('location', axis='columns')
df11.shape

#preparing X and y
X=df11.drop('price', axis='columns')
y=df11.price
y.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=10)


from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(X_train,y_train)

lr_clf.score(X_test, y_test)


#using k-fold cross validation to pick the best algorithm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv=ShuffleSplit(n_splits=5, test_size=0.2,random_state=0)
cross_val_score(LinearRegression(), X,y,cv=cv)

#trying different regression algorithms
#usinng grid search cv to figure out best score , which algorithms provides
from sklearn.model_selection import GridSearchCV 

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

#a function performing hyper parameter tuning
def find_bestmodel_using_gs_cv(X,y):
    algos={
        'linear regression':{
            'model':LinearRegression(),
            'params':{
                'fit_intercept':[True,False]
                }   
            },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random', 'cyclic']    
                }    
            },
        'decision tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
                }
            }
        
        
        }
    scores=[]
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name, config in algos.items():
        gs=GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model':algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
                })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
     

op=find_bestmodel_using_gs_cv(X, y)          

#defining an input for price prediction
def predict_price(location, sqft, bath, bhk):
    
    #finding column index
    loc_index=np.where(X.columns==location)[0][0]
    
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index >=0:
        x[loc_index]=1
        
    return lr_clf.predict([x])[0]

predict_price('1st Phase JP Nagar', 1000, 2, 2)

predict_price('1st Phase JP Nagar', 1000, 3, 3)


#ab testing???
#exporting to pickle
import pickle
with open(r'S:\satya\python\hpp\banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf, f)

#other than model , we also need columns information for prediction
#the way columns are indexed is important, export it to json
import json
columns={
    'data_columns':[col.lower() for col in X.columns]
}
with open(r'S:\satya\python\hpp\columns.json','w') as f:
    f.write(json.dumps(columns))


   
    
  

 



















    


