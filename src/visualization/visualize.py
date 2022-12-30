"""Charts and plots for analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats

class visualization():

    def EDA(df):
        """Exploratory data analysis for dataset."""
        warnings.filterwarnings('ignore')

        fig,ax = plt.subplots(6,3,figsize=(30,60))
        for index,i in enumerate(df.iloc[:,1:7]):
            sns.distplot(df[i],ax=ax[index,0])
            sns.boxplot(df[i],ax=ax[index,1])
            stats.probplot(df[i],plot=ax[index,2])

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        plt.suptitle("Visualizing Continuous Columns",fontsize=40)
