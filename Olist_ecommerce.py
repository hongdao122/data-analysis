import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
%matplotlib inline
import os
import sqlite3
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set(style="ticks")
import gc
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")
pd.set_option('display.max_columns', 100)
np.random.seed(42)
import plotly
from datetime import datetime, timedelta
import plotly.offline as pyoff
import plotly.graph_objs as go
#initiate visualization library for jupyter notebook 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
pyoff.init_notebook_mode(connected=True)
%matplotlib inline

from google.colab import drive
drive.mount('/content/gdrive')

customer = pd.DataFrame(pd.read_csv('/content/gdrive/MyDrive/OM - Data Analyst/olist_customers_dataset.csv'))
geo = pd.DataFrame(pd.read_csv('/content/gdrive/MyDrive/OM - Data Analyst/olist_geolocation_dataset.csv'))
order_item = pd.DataFrame(pd.read_csv('/content/gdrive/MyDrive/OM - Data Analyst/olist_order_items_dataset.csv'))
order_payment =  pd.DataFrame(pd.read_csv('/content/gdrive/MyDrive/OM - Data Analyst/olist_order_payments_dataset.csv'))
order_review = pd.DataFrame(pd.read_csv('/content/gdrive/MyDrive/OM - Data Analyst/olist_order_reviews_dataset.csv'))
order_data = pd.DataFrame(pd.read_csv('/content/gdrive/MyDrive/OM - Data Analyst/olist_orders_dataset.csv'))
product_data = pd.DataFrame(pd.read_csv('/content/gdrive/MyDrive/OM - Data Analyst/olist_products_dataset.csv'))
sellers = pd.DataFrame(pd.read_csv('/content/gdrive/MyDrive/OM - Data Analyst/olist_sellers_dataset.csv'))
product_cat = pd.DataFrame(pd.read_csv('/content/gdrive/MyDrive/OM - Data Analyst/product_category_name_translation.csv'))

#defining visualizaition functions
def format_spines(ax, right_border=True):
    
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')
    ax.spines['top'].set_visible(False)
    if right_border:
        ax.spines['right'].set_color('#FFFFFF')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')
    

def count_plot(feature, df, colors='Blues_d', hue=False, ax=None, title=''):
    
    # Preparing variables
    ncount = len(df)
    if hue != False:
        ax = sns.countplot(x=feature, data=df, palette=colors, hue=hue, ax=ax)
    else:
        ax = sns.countplot(x=feature, data=df, palette=colors, ax=ax)
        
    format_spines(ax)

    # Setting percentage
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
    
    # Final configuration
    if not hue:
        ax.set_title(df[feature].describe().name + ' Analysis', size=13, pad=15)
    else:
        ax.set_title(df[feature].describe().name + ' Analysis by ' + hue, size=13, pad=15)  
    if title != '':
        ax.set_title(title)       
    plt.tight_layout()
    
    
def bar_plot(x, y, df, colors='Greens_d', hue=False, ax=None, value=False, title=''):
    
    # Preparing variables
    try:
        ncount = sum(df[y])
    except:
        ncount = sum(df[x])
    #fig, ax = plt.subplots()
    if hue != False:
        ax = sns.barplot(x=x, y=y, data=df, palette=colors, hue=hue, ax=ax, ci=None)
    else:
        ax = sns.barplot(x=x, y=y, data=df, palette=colors, ax=ax, ci=None)

    # Setting borders
    format_spines(ax)

    # Setting percentage
    for p in ax.patches:
        xp=p.get_bbox().get_points()[:,0]
        yp=p.get_bbox().get_points()[1,1]
        if value:
            ax.annotate('{:.2f}k'.format(yp/1000), (xp.mean(), yp), 
                    ha='center', va='bottom') # set the alignment of the text
        else:
            ax.annotate('{:.1f}%'.format(100.*yp/ncount), (xp.mean(), yp), 
                    ha='center', va='bottom') # set the alignment of the text
    if not hue:
        ax.set_title(df[x].describe().name + ' Analysis', size=12, pad=15)
    else:
        ax.set_title(df[x].describe().name + ' Analysis by ' + hue, size=12, pad=15)
    if title != '':
        ax.set_title(title)  
    plt.tight_layout()

# converting date columns to datetime
date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_columns:
    order_data[col] = pd.to_datetime(order_data[col], format='%Y-%m-%d %H:%M:%S')

# cleaning up name columns
customer['customer_city'] = customer['customer_city'].str.title()
order_payment['payment_type'] = order_payment['payment_type'].str.replace('_', ' ').str.title()
# engineering new/essential columns
order_data['delivery_against_estimated'] = (order_data['order_estimated_delivery_date'] - order_data['order_delivered_customer_date']).dt.days
order_data['order_purchase_year'] = order_data.order_purchase_timestamp.apply(lambda x: x.year)
order_data['order_purchase_month'] = order_data.order_purchase_timestamp.apply(lambda x: x.month)
order_data['order_purchase_dayofweek'] = order_data.order_purchase_timestamp.apply(lambda x: x.dayofweek)
order_data['order_purchase_hour'] = order_data.order_purchase_timestamp.apply(lambda x: x.hour)
order_data['order_purchase_day'] = order_data['order_purchase_dayofweek'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
order_data['order_purchase_mon'] = order_data.order_purchase_timestamp.apply(lambda x: x.month).map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
# Changing the month attribute for correct ordenation
order_data['month_year'] = order_data['order_purchase_month'].astype(str).apply(lambda x: '0' + x if len(x) == 1 else x)
order_data['month_year'] = order_data['order_purchase_year'].astype(str) + '-' + order_data['month_year'].astype(str)
#creating year month column
order_data['month_year'] = order_data['order_purchase_timestamp'].map(lambda date: 100*date.year + date.month)

# make master dataset for further analysis

# group payment value
order_payment_sum = order_payment.groupby(['order_id'])['payment_value'].sum()
df = pd.merge(order_data,order_payment_sum, on ='order_id')
master_1 = pd.merge(df, customer, on = 'customer_id')
master_1

# make master dataset of product order = ed
product = pd.merge(product_data, product_cat, on = 'product_category_name')
df_order_item = pd.merge(order_item, product, on = 'product_id')
order_customer = pd.merge(order_data[['order_id','customer_id']], customer , on = 'customer_id')
df1_order_item = pd.merge(df_order_item, order_customer, on = 'order_id')
master_order_item = pd.merge(df1_order_item, sellers, on = 'seller_id')
master_order_item

# displaying missing value counts and corresponding percentage against total observations
missing_values = df.isnull().sum().sort_values(ascending = False)
percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
pd.concat([missing_values, percentage], axis=1, keys=['Values', 'Percentage']).transpose()

df_revenue_mom = master_1.groupby(['month_year'])['payment_value'].sum().reset_index()
df_revenue_mom = df_revenue_mom.query("month_year != [201609,201610,201612,201701,201809, 201810]")

fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted', color_codes=True, style='darkgrid')
bar_plot(x='month_year', y='payment_value', df=df_revenue_mom, value = True)
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.set_title("Revenue Trend", fontsize = 20)

ax.tick_params(axis='x', labelrotation=45)
plt.show()

#calculating for monthly revenie growth rate
# using pct_change() function to see monthly percentage change
df_revenue_mom['MonthlyGrowth'] = df_revenue_mom['payment_value'].pct_change()
df_revenue_growth = df_revenue_mom[['month_year','MonthlyGrowth']]
df_revenue_growth = df_revenue_growth.query("month_year != [201609,201610,201612,201701,201809, 201810]")
df_revenue_growth

fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted', color_codes=True, style='darkgrid')
bar_plot(x='month_year', y='MonthlyGrowth', df=df_revenue_growth)
ax.set_xlabel("Month")
ax.set_ylabel("% Revenue Change")
ax.set_title("Revenue Change MoM", fontsize = 20)

ax.tick_params(axis='x', labelrotation=45)
plt.show()

#creating monthly active customers dataframe by counting unique Customer IDs
df_customer_active = master_1.groupby('month_year')['customer_unique_id'].nunique().reset_index()
df_customer_active = df_customer_active.query("month_year != [201609,201610,201612,201701,201809, 201810]")


fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted', color_codes=True, style='whitegrid')
bar_plot(x='month_year', y='customer_unique_id', df=df_customer_active, value=True)

ax.set_xlabel("Month")
ax.set_ylabel("Active Customer")
ax.set_title("Active Customer MoM", fontsize = 20)

ax.tick_params(axis='x', labelrotation=45)
plt.show()

#creating monthly active customers dataframe by counting unique Customer IDs
df_monthly_sales = master_1.groupby('month_year')['order_status'].count().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted', color_codes=True, style='whitegrid')
bar_plot(x='month_year', y='order_status', df=df_monthly_sales, value=True)
ax.tick_params(axis='x', labelrotation=90)
plt.show()

order_usual = order_item.groupby(['order_id'])['order_item_id'].max().reset_index()
order_usual = order_usual['order_item_id'].value_counts()
order_usual.head()
plt.figure(figsize=(8,8))
ax=sns.barplot(x=order_usual.index,y=order_usual.values,color="green")
ax.set_xlabel("Item")
ax.set_ylabel("Number of Transaction")
ax.set_title("Number of Item per Transaction", fontsize = 20)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0)

product_ordered = master_order_item['product_id'].value_counts()
product_ordered.columns = ['product_id','counts']
product_ordered

# Top 10 products most purchased

temp1 = master_order_item.groupby(['product_category_name_english'])['order_id'].count().reset_index()
top_10 = temp1.sort_values(by=['order_id'], ascending = False).head(10)


ax = sns.barplot(x="order_id", y="product_category_name_english", data=top_10, palette="Blues_d")
ax.set_title("Total Orders of Product Category", fontsize = 15)
ax.tick_params(axis='x', labelrotation=0)
plt.show()

# Top 10 products by total revenue

temp2 = master_order_item.groupby(['product_category_name_english'])['price'].sum().reset_index()
top_10_by_revenue = temp2.sort_values(by=['price'], ascending = False).head(10)

ax = sns.barplot(x="price", y="product_category_name_english", data=top_10_by_revenue, palette="Blues_d")
ax.set_title("Revenue of Product Category", fontsize = 15)
ax.tick_params(axis='x', labelrotation=0)
plt.show()


weekday = master_1['order_purchase_timestamp'].dt.weekday
hour = master_1['order_purchase_timestamp'].dt.hour
pprice = master_1['payment_value']

purchase = pd.DataFrame({'day of week': weekday, 'hour': hour, 'price': pprice})
purchase['day of week'] = purchase['day of week'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
purchase.head()

purchase_count = purchase.groupby(['day of week', 'hour']).count()['price'].unstack()
plt.figure(figsize=(16,6))
sns.heatmap(purchase_count.reindex(index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']), 
            cmap="YlGnBu", annot=True, fmt="d", linewidths=0.5)

total = len(master_1)
plt.figure(figsize=(16,12))

# plt.suptitle('CUSTOMER State Distributions', fontsize=22)

plt.subplot(212)
g = sns.countplot(x='customer_state', data=master_1)
g.set_title("Customer's State Distribution", fontsize=20)
g.set_xlabel("State", fontsize=17)
g.set_ylabel("Count", fontsize=17)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
sizes = []
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}%'.format(height/total*100),
            ha="center", fontsize=12) 
g.set_ylim(0, max(sizes) * 1.1)
plt.show()

total_order = master_order_item['order_id'].count()
total_order
total_delivery = master_order_item.groupby(['seller_state']).agg({'order_id':'count'}).rename(columns = {'order_id':'total delivery'}).reset_index().sort_values(by= 'total delivery', ascending = False)
total_delivery
total_delivery['percent'] = total_delivery['total delivery']/total_order
total_delivery = total_delivery.loc[total_delivery['total delivery'] != 0]
total_delivery


fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted',color_codes= True, style='darkgrid')
bar_plot(x='seller_state', y='total delivery', df=total_delivery)
ax.set_xlabel("State",fontsize=17)
ax.set_ylabel("Delivery ",fontsize=17)
ax.set_title("Seller's State Distribution",fontsize=20)
ax.tick_params(axis='x')
plt.show()

total_payment = order_payment['payment_value'].sum()
total_payment
temp12 = order_payment.groupby(['payment_type'])['payment_value'].sum().reset_index().sort_values(by= 'payment_value', ascending = False)
temp12['percent'] = temp12['payment_value']/total_payment
temp12 = temp12.loc[temp12['payment_value'] != 0]
temp12


fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted',color_codes= True, style='darkgrid')
bar_plot(x='payment_type', y='percent', df=temp12)
ax.set_xlabel("Type of Payment",fontsize=17)
ax.set_ylabel("Number of Payment",fontsize=17)
ax.set_title("Number of Payment by Type",fontsize=20)
ax.tick_params(axis='x')
plt.show()

# Seting regions
sudeste = ['SP', 'RJ', 'ES','MG']
nordeste= ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA']
norte =  ['AM', 'RR', 'AP', 'PA', 'TO', 'RO', 'AC']
centro_oeste = ['MT', 'GO', 'MS' ,'DF' ]
sul = ['SC', 'RS', 'PR']
master_order_item.loc[master_order_item['customer_state'].isin(sudeste), 'cust_Region'] = 'Southeast'
master_order_item.loc[master_order_item['customer_state'].isin(nordeste), 'cust_Region'] = 'Northeast'
master_order_item.loc[master_order_item['customer_state'].isin(norte), 'cust_Region'] = 'North'
master_order_item.loc[master_order_item['customer_state'].isin(centro_oeste), 'cust_Region'] = 'Midwest'
master_order_item.loc[master_order_item['customer_state'].isin(sul), 'cust_Region'] = 'South'
freight_cost = master_order_item[['customer_state','cust_Region','freight_value']]

freight_cost = master_order_item.groupby(['seller_state','cust_Region']).agg({'freight_value':'mean'}).reset_index()
freight_cost = freight_cost.pivot('seller_state','cust_Region','freight_value')
freight_cost
plt.figure(figsize=(15,8))
ax=sns.heatmap(freight_cost,annot=True,cmap="OrRd")
ax.set_xlabel("Customer Region", fontsize =17 )
ax.set_ylabel("Seller State",fontsize =17 )
ax.set_title("Heatmap of freight cost by seller and customer location",fontsize =20)
ax.tick_params(axis = 'y',labelrotation=0)

# round(pd.crosstab(['order_item_id'], order_review['review_score'], normalize='index') *100,2)[:12].T
review_avg = order_review.groupby(['order_id']).agg({'review_score':'max'}).reset_index()
review = pd.merge(review_avg, master_order_item, on = 'order_id')

pivot = round(pd.crosstab(review['order_item_id'], review['review_score'], normalize='index') *100,2)[:12].T

plt.figure(figsize=(15,8))
ax=sns.heatmap(pivot,annot=True,cmap="OrRd")
ax.set_xlabel("Quantity per Order", fontsize =17 )
ax.set_ylabel("Review Score",fontsize =17 )
ax.set_title("Review score by Order Quantity (%)",fontsize =20)
ax.tick_params(axis = 'y',labelrotation=0)

list_product = list(top_10_by_revenue['product_category_name_english'])

list_product
review = order_review.groupby(['order_id'])['review_score'].mean().reset_index()
review_product = pd.merge(review, master_order_item,  on = 'order_id')[['product_category_name_english','review_score']]
review_product = review_product.groupby(['product_category_name_english']).agg({'review_score':['count','mean']}).reset_index()
review_product.columns = ['product_category_name_english','count','mean']


review_product_2 = review_product.loc[review_product['product_category_name_english'].isin(list_product)]
review_product_2.columns = ['product_category_name_english','count','mean']
review_product_2 = review_product_2.sort_values(by = ['mean'], ascending = False)
review_product_2

ax = sns.barplot(x="mean", y="product_category_name_english", data=review_product_2)
ax.set_title('Categories Review Score')
ax.set_xlabel('Average Score')
ax.set_ylabel('Product Category')


