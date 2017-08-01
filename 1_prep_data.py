import numpy as np
import pandas as pd
import datetime
import Tkinter
import tkFileDialog
import os.path
from sklearn import preprocessing

#-----------------------------------------------------------------------------------------
FABRIC_FILE = 'fabric_newtest.csv'
#fabric file is an automatically generated list of the parameter configuration 
#for each Run. This allows user to experiment with different configurations 
#for different datasets and find the optimal solution.
#-----------------------------------------------------------------------------------------


def File_Dialog(file_name):
# Opens a tkinter GUI that allows user to find the necessary documents in other directories if the default file sets specified above are not found.
    root = Tkinter.Tk()
    root.withdraw()

    print "The "+file_name+" data file was not found or is not properly formatted."
    print "Please locate a valid file manually."
    return tkFileDialog.askopenfilename(defaultextension="csv",parent=root,title = "Please select "+file_name+" file:")
    


def Get_Factory_Data():
# Opens default files specified above (or prompts user to locate files) and 
#returns the data frames within those files after some pre-processing 
#(ie. parsing dates, filtering out industries than companies, and replacing "
# " with "_" in data sets containing phrases.) 
    if os.path.isfile(FABRIC_FILE):
        dat_setup = pd.read_csv(FABRIC_FILE, parse_dates = [1,2])
    else:
        dat_setup = File_Dialog("fabric file")
    
    par_list = dat_setup.D_PARAM_LIST[3].split('|')
    inc_check = par_list[0]
    del par_list[0]

   
	if os.path.isfile('firms.csv'):
        dat_meta = pd.read_csv('firms.csv', parse_dates = [5])
    else:
        dat_meta = File_Dialog("firm meta data")
    
    dat_meta.Industry = map(lambda x: str.replace(x,' ','_'), dat_meta.Industry)
        

    if os.path.isfile('former.csv'):
        dat = pd.read_csv('former.csv', parse_dates = [2,3,4,5])
    else:
        dat = File_dialog("fundamental finance data")
    

    if inc_check == "inc":
        dat = dat.filter(items=par_list)
    elif inc_check == "exc":
        dat = dat.drop(par_list,1)

    return [dat_setup, dat_meta, dat]



def Filter_by_List(dat, col_name, f_list_ls = None, f_list_str = None):
# Takes in input of a string with elements separated by | character (ie. string-lists)
#string-lists must have a flag as their first element indicating whether the elements 
#in the list are to be excluded or included from analysis. If the input is an actual
#list, the elements automatically are set to be included.
#string-lists have a '|' as the delimiting character to keep the list as one element
#in csv elements within a string-list shouldn't have spaces, only underscores, 
#in order to avoid issues with split.
    if f_list_ls == None:
        if f_list_str == None:
            print "error in Filter_by_List: no list or string-list was detected"
        else:
           f_list = f_list_str.split('|')
            inc_check = f_list[0]
            del f_list[0]
    else:
        f_list = f_list_ls
        inc_check = "inc" 
    

    if inc_check == "inc":
        dat_mask = dat[col_name].isin(f_list)
    elif inc_check == "exc":
        dat_mask = ~dat[col_name].isin(f_list)
    else:
        print "error in Filter_by_List: no inclusion/exclusion flag was detected in string-list"
        return dat

    return dat.loc[dat_mask]



def Filter_by_Daterng(dat,col_name, start_date, end_date):
# Filters the data set by the data range specified in the fabric file.
    dt_mask = (dat[col_name]>start_date)&(dat[col_name]<end_date)
    return dat.loc[dt_mask]



def Filter_by_Threshold(dat,col_name,th_value, keep_elem_below = None):
# Generic function to filter out data points with specified attributes above 
#or below threshold. (keep_elem_below is an optional bool spec to indicate whether 
#to filter out elements with values below threshold; by default, it will keep
#elements over threshold.)
    if keep_elem_below:
        th_mask = dat[col_name] < th_value
    else:
        th_mask = dat[col_name] >= th_value
    
    
    return dat.loc[th_mask]



def Fill_Dataset_Nas(dat):
# Fills in dataset based on the forward fill algorithm. If there was no previous
#value to forward fill the na, it will set value to 0.
    full_dat = pd.DataFrame   
    
    for tk in list(dat.ticker.value_counts().index):
        tk_mask = (dat.ticker == tk)
        temp = dat.loc[tk_mask]
        
        temp = temp.fillna(method = 'ffill')
        temp = temp.fillna(0)
        
        if full_dat.empty:
            full_dat = temp
        else:
            full_dat = full_dat.append (temp, ignore_index = True)
            
    return full_dat



def Normalize_Data_Set(dat,n):
# Normalizes data sets after converting dates to numeric values.
	cnt = 0.0

	key = pd.DataFrame(columns = ["ticker","rank"])
	key["ticker"] = dat["ticker"].value_counts().index
	key["rank"] = range(0, len(key))

	key.to_csv("flora"+str(n)+".csv", index = False, index_label = False)
	for i in list(dat["ticker"].value_counts().index):
		dat["ticker"].loc[dat["ticker"] == i] = cnt
		cnt+= 1.0

	dat["datekey"] = pd.to_numeric(dat["datekey"])    

	x = dat.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	dat_norm = pd.DataFrame(x_scaled, columns = list(dat))

	return dat_norm


# In[11]:

def Shape_Data_Set(dat, n_seq_size):
# Shapes data based on the sequence lenth specified by user in fabric file.
    n_total_tk = len(dat.ticker.value_counts().index)
    n_total_pnts = len(dat)
    n_seq = 0
    
	shaped_dat = pd.DataFrame

    for i in range(0, n_total_pnts-n_seq_size):
		temp = dat[i:i+n_seq_size]
        
        if (len(temp.ticker.value_counts()) == 1) & (len(temp) == n_seq_size):
            n_seq += 1
            temp['seq_num'] = np.full((n_seq_size,1),n_seq)
            temp['n_in_seq'] = range(0,n_seq_size)
            

            if shaped_dat.empty:
                shaped_dat = temp
            else:
                shaped_dat = shaped_dat.append(temp, ignore_index = True)
            
	return shaped_dat


#MAIN#################################################################################

dat_setup, dat_meta, dat = Get_Factory_Data()

for n in range(0,len(dat_setup)):
	dat_mat = Filter_by_List(dat_meta, "Sector", f_list_str = dat_setup.D_SECTOR_LIST[n])
	dat_hat = Filter_by_List(dat_mat, "Industry", f_list_str = dat_setup.D_INDUSTRY_LIST[n])
	dat_cat = Filter_by_List(dat_hat, "Ticker", f_list_str = dat_setup.D_TKEXCLUDE_LIST[n])

	dat_dat = Filter_by_List(dat,"ticker", f_list_ls = list(dat_cat.Ticker))
	dat_bat = Filter_by_Daterng(dat_dat,"datekey", dat_setup.D_START_DATE[n], dat_setup.D_END_DATE[n])
	dat_fat = Filter_by_Threshold(dat_bat,"marketcap",dat_setup.D_MIN_MARKETCAP[n])
	tk_f_dtpnt = list(dat_fat.ticker.value_counts().loc[dat_fat.ticker.value_counts()>=dat_setup.D_MIN_DATAPNTS[n]].index)
	dat_flat = Filter_by_List(dat_fat,"ticker",f_list_ls = tk_f_dtpnt)

	dat_f = Fill_Dataset_Nas(dat_flat)
	dat_s = Shape_Data_Set(dat_f,dat_setup.N_SEQ_SIZE[n])
	dat_n = Normalize_Data_Set(dat_s,n)


	dat_n.to_csv("forged"+str(n)+".csv",index = False, index_label = False)

######################################################################################


