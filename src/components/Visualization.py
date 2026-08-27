import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config.paths import CHARTS_PATH


def save_box_chart(series,title,filename,xlabel='',ylabel=''):
    plt.figure(figsize=(10,10))
    plt.boxplot(series.values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(CHARTS_PATH,filename))
    plt.close()
    plt.tight_layout()

def save_histogram(series,title,filename,bins=20):
    plt.figure(figsize=(10,10))
    plt.hist(series,bins=bins)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH,filename))
    plt.close()

def save_bar_chart(series,title,filename,xlabel='',ylabel='Count'):
    plt.figure(figsize=(10,10))
    plt.bar(series.value_counts().index.astype(str),series.value_counts().values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH,filename))
    plt.close()



def save_heatmap(crosstab,title,filename):
    plt.figure(figsize=(10,10))

    sns.heatmap(crosstab,annot=True,cmap='coolwarm')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH,filename))
    plt.close()

def create_visuals(data:pd.DataFrame):
    save_box_chart(data['Age'],
                   title='Age Distribution',
                   filename='Age_distribution.png')



    save_histogram(data['Annual_Premium'],
                   title='Annual premium of distribution',
                   filename='histogram_of_Annual_premium.png')

    save_bar_chart(data['Response'],
                   title='Response',
                   filename='Response_bar_chart_distribution.png')

    save_heatmap(pd.crosstab(data['Vehicle_Age'],data['Vehicle_Damage']),
                 title='Vehicle Age vs Vehicle Damage',
                 filename='Agevsdamage_distribution.png')



def main():
    from src.components.data_ingestion import load_data
    data=load_data()
    create_visuals(data)

if __name__=='__main__':
    main()
