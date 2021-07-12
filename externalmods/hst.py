import os, sys, glob, subprocess
import numpy as np
import math
import pandas as pd
import time
import tarfile
import threading
from astropy.io import fits
# PLOTTING
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
mpl.style.use('seaborn-bright')
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 12}
mpl.rc('font', **font)
import seaborn as sns
sns.set_style('whitegrid')
#ignore pink warnings
import warnings
warnings.filterwarnings('ignore')
# Allow for large # columns
pd.set_option('display.max_columns', 0)
import datetime as dt
import tzlocal as tz
# SCIPY / statsmodels
import scipy.stats as stats
# D'Agostino and Pearson's omnibus test
from scipy.stats import normaltest as normtest
from scipy import interp
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import scipy.stats as stats
from scipy.stats import normaltest as normtest # D'Agostino and Pearson's omnibus test
from collections import Counter
# Pre-processing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# modeling
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xg
# NNs
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
# eval
from sklearn import metrics
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score
from sklearn.metrics import average_precision_score, precision_score, f1_score


def pipe(*args, encoding="utf-8", print_output=False, raise_exception=False):
    """Every arg should be a subprocess command string which will be run and piped to
    any subsequent args in a linear process chain.  Each arg will be split into command
    words based on whitespace so whitespace embedded within words is not possible.

    Returns stdout from the chain.
    """
    pipes = []
    for cmd in args:
        words = cmd.split()
        if pipes:
            p = subprocess.Popen(words, stdin=pipes[-1].stdout, stdout=subprocess.PIPE)
            pipes[-1].stdout.close()
        else:
            p = subprocess.Popen(words, stdout=subprocess.PIPE)
        pipes.append(p)
    output = p.communicate()[0]
    ret_code = p.wait()
    if ret_code and raise_exception:
        raise RuntimeError(f"Subprocess failed with with status: {ret_code}")
    output = output.decode(encoding) if encoding else output
    if print_output:
        print(output, end="")
    return output

def parse_results(results):
    """Break up multiline output from ls into a list of (name, size) tuples."""
    return [(line.split()[8], int(line.split()[4])) for line in results.splitlines() if line.strip()]
    #return [(line.split()[1], int(line.split()[0])) for line in results.splitlines() if line.strip()]


def file_tree(startpath):
    cwd = os.getcwd()
    os.chdir(startpath)
    for root, dirs, files in os.walk(os.getcwd()):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in sorted(files, key=lambda f: os.path.getsize(root + os.sep + f)):
            #print('{} - {}'.format(f, os.path.getsize(root + os.sep + f)))
            print('{}{} - {}'.format(subindent, f, os.path.getsize(root + os.sep + f)))
    os.chdir(cwd)    
    
def total_size(ls_out):
    out = parse_results(ls_out)
    file_dict = {}
    for (name, size) in out:
        filename = name
        file_dict[filename] = size
    
    byte = 0
    for k, v in file_dict.items():
        byte += v
    mbsize = byte*10**-6
    return mbsize
    
    
def list_objects(path):
    object_dict = {}
    output = pipe(f"aws s3 ls --recursive {path}")
    results = [(int(line.split()[2]), line.split()[3]) for line in output.splitlines() if line.strip()]
    for (size, name) in results:
        filename = os.path.basename(name)
        if size > 0:
            object_dict[filename] = size
    return object_dict

    
def list_files(startpath, ipppssoot):
    file_dict = {}
    for root, _, files in os.walk(startpath):
        for f in sorted(files, key=lambda f: os.path.getsize(root + os.sep + f)):
            if f.startswith(ipppssoot[0:5].lower()):
                file_dict[f] = os.path.getsize(root + os.sep + f)
    return file_dict

def list_tarfiles(startpath):
    file_dict = {}
    for root, _, files in os.walk(startpath):
        for f in sorted(files, key=lambda f: os.path.getsize(root + os.sep + f)):
            if not f.endswith('DS_Store'):
                t = f.strip('.tar.gz')
                file_dict[t] = os.path.getsize(root + os.sep + f)
    return file_dict


def raw_inputs(ipst_dir):
    """
    Returns dict of input files for HST instruments
    startpath = ipst subfolder
    EX: make a dict of all raw files for each wfc3 ipppsoot
    inside a parent dir stored as a path object variable `ipsts_wfc3`'
    wfc3_raw = {} 
    for ipst_dir in ipsts_wfc3:
        ipst = str(ipst_dir).lower() # older runs had folders capitalized 
        wfc3_raw[ipst] = raw_inputs(ipst_dir, sfx_list=["asn", raw"])
    """
    # gets list of appropriate file suffices to look for 
    # this makes it possible to extract just the inputs from an outputs folder
    # where previews, calibrated fits files and log files are mixed in.
    instr = ipst_dir[0]
    if instr == 'a' or 'i': # acs, wfc3
        sfx_list = ["asn", "raw"]
    elif instr == 'l': # cos
        sfx_list == ["asn","raw","epc","spt","rawaccum","rawacq","rawtag","pha"]
    elif instr == 'o': # stis
        sfx_list == ["asn", "raw", "epc", "tag", "wav"]
      
    file_dict = {}
    for root, _, files in os.walk(ipst_dir):
        for f in sorted(files, key=lambda f: os.path.getsize(root + os.sep + f)):
            ext = f.split('.')[-1]
            if ext == 'fits':
                sfx = f.split('_')[-1]
                for s in sfx_list:
                    if s in sfx:
                        file_dict[f] = os.path.getsize(root + os.sep + f)
    return file_dict

def input_stats(file_dict):

    filestats = {}
    n_files = 0
    total_mb = 0

    for name, size in file_dict.items():
        n_files += 1
        size_mb = size*10**-6
        total_mb += size_mb
        
    filestats["n_files"] = n_files
    filestats["total_mb"] = np.round(total_mb, 1)

    return filestats

def remove_missing_data(raw_dict):
    raw_dict2 = {}
    for k, v in raw_dict.items():
        if len(v) > 0:
            raw_dict2[k] = v
    print(len(raw_dict) - len(raw_dict2))
    return raw_dict2

def fits_keywords(fits_file):
    """
    'DETECTOR':
        * WFC3 = IR (1024 x 1024) | UVIS (mosaic of two 4096 x 2051 px)
        * ACS = WFC (mosaic of two 2048 x 4096 px) | HRC or SBC (1024 x 1024)
    'SUBARRAY': (BOOL) if True, will process faster bc readouts = smaller than full-frame images
    'DRIZCORR': PERFORM or OMIT (run drizzling step)
    'PCTECORR': PERFORM (versus OMIT)
    'NAXIS1'/'NAXIS2' : size of images
    'CRSPLIT': > 1 = multiple imgs for comic ray rejection
    Not using due to rarity of occurrence: 'RPTOBS': repeated obs at same coords to increase SNR (>1 img) 
    """
    keys = {}
    with fits.open(fits_file) as hdul:
        keywords = ['DETECTOR', 'SUBARRAY', 'DRIZCORR', 
                    'PCTECORR', 'CRSPLIT']
        h0 = hdul[0].header
        h1 = hdul[1].header
        for k in keywords:
            try:
                keys[k] = h0[k]
            except KeyError:
                try:
                    keys[k] = h1[k]
                except KeyError:
                    continue

    return keys

def get_keys(file_dict):
    instr_dir = os.getcwd()
    ipst_keywords = {}
    for ipst, files in file_dict.items():
        ipst_dir = ipst.upper()
        os.chdir(ipst_dir)
        keyword_list = []
        for f in files:
            keys = fits_keywords(f)
            if len(keys)> 0:
                keyword_list.append(fits_keywords(f))
        ipst_keywords[ipst] = keyword_list
        os.chdir(instr_dir)
    return ipst_keywords


def strip_keys(ipst_keywords):
    KEYS = {}
    for key, lists in ipst_keywords.items():
        ln = []
        for dct in lists:
            ln.append(len(dct))
        for index, dct in enumerate(lists):
            if len(dct) == max(ln):
                KEYS[key] = dct
    return KEYS

def scrub_keys(data):
    # copy columns to validate before making changes
    df = data.copy()
    cols = df.columns
    # DRIZCORR
    if 'DRIZCORR' in cols:
        df['drizcorr'] = df['DRIZCORR']
        df['drizcorr'].loc[(df['drizcorr'].isna()) | (df['drizcorr'] == 'OMIT')] = 0
        df['drizcorr'].loc[(df['drizcorr'] == 'PERFORM') | (df['drizcorr'] == 'COMPLETE')] = 1
        df['drizcorr'] = df['drizcorr'].astype('int64')
        df.drop(columns=['DRIZCORR'], axis=1, inplace=True)
    # PCTECORR
    if 'PCTECORR' in cols:
        df['pctecorr'] = df['PCTECORR']
        df['pctecorr'].loc[(df['pctecorr'].isna()) | (df['pctecorr'] == 'OMIT')] = 0
        df['pctecorr'].loc[df['pctecorr'] == 'PERFORM'] = 1
        df['pctecorr'] = df['pctecorr'].astype('int64')
        df.drop(columns=['PCTECORR'], axis=1, inplace=True)
    # CRSPLIT  
    if 'CRSPLIT' in cols:
        df['crsplit'] = df['CRSPLIT']
        df['crsplit'].loc[(df['crsplit'].isna()) | (df['crsplit'] == 0)] = 0
        df['crsplit'].loc[df['crsplit'] == 1] = 1
        df['crsplit'].loc[df['crsplit'] > 1] = 2
        df['crsplit'] = df['crsplit'].astype('int64')
        df.drop(columns=['CRSPLIT'], axis=1, inplace=True)
    # SUBARRAY
    if 'SUBARRAY' in cols:
        df['subarray'] = df['SUBARRAY']
        # convert boolean to integer
        df['subarray'].loc[df['subarray'] == True] = 1
        df['subarray'].loc[(df['subarray'] == False) | (df['subarray'].isna())] = 0
        df['subarray'] = df['subarray'].astype('int64')
        df.drop(columns=['SUBARRAY'], axis=1, inplace=True)
    # DETECTOR
    if 'DETECTOR' in cols:
        df['detector'] = df['DETECTOR']
        df['detector'].loc[(df['detector'] == 'UVIS') | 
                           (df['detector'] == 'WFC')] = 1
        df['detector'].loc[(df['detector'] == 'IR') | (df['detector'] == 'HRC') | 
                           (df['detector'] == 'SBC') | (df['detector'] == 'NUV') | 
                           (df['detector'] == 'FUV') | (df['detector'] == 'CCD') |
                           (df['detector'] == 'FUV-MAMA') | (df['detector'] == 'NUV-MAMA') |
                           (df['detector'].isna())] = 0
        df['detector'] = df['detector'].astype('int64')
        df.drop(columns=['DETECTOR'], axis=1, inplace=True)
    if 'ipst' in cols:
        df.drop(columns=['ipst'], axis=1, inplace=True)
    mem = df['memory']
    wall = df['wallclock']
    df.drop(columns=['wallclock', 'memory'], axis=1, inplace=True)
    df['wallclock'] = wall
    df['memory'] = mem
    return df

def file_stats(file_dict):

    filestats = {}
    n_files = 0
    total_mb = 0

    for name, size in file_dict.items():
        n_files += 1
        size_mb = size*10**-6
        total_mb += size_mb

        suffix = name.split('_')[1:]
        sfx = '_'.join(suffix[:])
        if len(sfx) > 0:
            if sfx not in list(filestats.keys()):
                filestats[sfx] = 1
            else:
                filestats[sfx] += 1
        
    filestats["n_files"] = n_files
    filestats["total_mb"] = np.round(total_mb, 1)

    return filestats


def parse_logs(ipsts):
    
    cal_metrics = {}

    for ipst in ipsts:
        ipst_dir = os.path.abspath(ipst)
        proc_log = os.path.join(ipst_dir, "logs", "process_metrics.txt")
        prev_log = os.path.join(ipst_dir, "logs", "preview_metrics.txt")
        
        logfiles = [proc_log, prev_log]
        
        metrics = {'wallclock':0.0, 'memory':0.0}
        clockstring = []
        memstring = []
        for log in logfiles:
            with open(log) as f:
                text = [line.strip('\t').strip('\n') for line in f.readlines()]
                
                for index, string in enumerate(text):
#                     if index == 3: # % cpu
#                         pct = string.replace('Percent of CPU this job got: ', '').strip('%')
#                         cpu = math.ceil(int(pct)/100)

                    if index == 4: # elapsed wallclock time
                        timestamp  = string.replace('Elapsed (wall clock) time (h:mm:ss or m:ss): ', '')
                        secs = float(timestamp.split(':')[0])*60 + float(timestamp.split(':')[-1])
                        secstr = str("{:0.2f}").format(secs)
                        clockstring.append(secstr)

                    elif index == 9: # ram
                        kb = np.float(string.replace('Maximum resident set size (kbytes): ', ''))
                        mb = np.float(kb/1000)
                        mbstr = str("{:0.2f}").format(mb)
                        memstring.append(mbstr)
        
        metrics["wallclock"] = np.round(float(clockstring[0]) + float(clockstring[1]), 2)
        metrics["memory"] = np.round(float(memstring[0]) + float(memstring[1]), 2)
        cal_metrics[ipst] = metrics
    return cal_metrics


def parse_metrics(ipsts):
    
    cal_metrics = {}
    file_dict = {}
    file_types = []
    memstats = ['n_files', 'total_mb', 'wallclock', 'memory']
    
    for ipst in ipsts:
        ipst_dir = os.path.abspath(ipst)
        proc_log = os.path.join(ipst_dir, "logs", "process_metrics.txt")
        prev_log = os.path.join(ipst_dir, "logs", "preview_metrics.txt")
        logfiles = [proc_log, prev_log]
        
        # log file calibration metrics
        metrics = {"wallclock":0.0, "memory":0.0}
        clockstring = []
        memstring = []
        for log in logfiles:
            try:
                with open(log) as f:
                    text = [line.strip('\t').strip('\n') for line in f.readlines()]
                    for string in text:
                        if string.startswith('Elapsed (wall clock)'):
                            timestamp  = string.replace("Elapsed (wall clock) time (h:mm:ss or m:ss): ", '')
                            secs = float(timestamp.split(":")[0])*60 + float(timestamp.split(":")[-1])
                            secstr = str("{:0.2f}").format(secs)
                            clockstring.append(secstr)
                        elif string.startswith('Maximum resident set size'):
                            kb = np.float(string.replace("Maximum resident set size (kbytes): ", ''))
                            mb = np.float(kb/1000)
                            mbstr = str("{:0.2f}").format(mb)
                            memstring.append(mbstr)
            except FileNotFoundError:
                clockstring = ['0', '0']
                memstring = ['0', '0']
                        
        metrics["wallclock"] = np.round(float(clockstring[0]) + float(clockstring[1]), 2)
        metrics["memory"] = np.round(float(memstring[0]) + float(memstring[1]), 2)
        
        # output file metrics
        file_dict = list_files(ipst_dir, ipst)
        stats = file_stats(file_dict)
        
        for k,v in stats.items():
            if k in memstats:
                continue
            elif k not in file_types:
                file_types.append(k)
            
        stats.update(metrics)

        cal_metrics[ipst] = stats
        
    return cal_metrics




def make_lower(data):
    index = list(data.index)
    index_lower = []
    for i in index:
        index_lower.append(i.lower())
    data['ipst'] = index_lower
    data.set_index('ipst', inplace=True, drop=True)
    return data



# this will change directory to location of raw input files

def find_fits(path_to_raw):
    os.chdir(path_to_raw)
    file_list = os.listdir(os.getcwd())
    for file in file_list:
        if file.startswith(".DS_"):
            file_list.remove(file)
    asn_files = list(glob.glob(f"*_asn.fits"))
    asn_records = {}
    for asn_file in asn_files:
        with fits.open(asn_file) as hdul:
            ipppssoot = hdul[0].header['ASN_ID']
            rec = hdul[1].data
        asn_records[ipppssoot] = [i.lower() for (i,r,b) in list(rec)]
        
    fits_files = {}
    for key,recs in asn_records.items():
        files = []
        for rec in recs:
            for f in file_list:
                if f.startswith(rec):
                    files.append(f) 
        fits_files[key] = files

    for asn_file in asn_files:
        key = asn_file[0:9]
        if asn_file not in fits_files[key]:
            fits_files[key].append(asn_file)

    reduced_list = file_list
    
    for ipppssoot, files in list(fits_files.items()):
        if ipppssoot.endswith("0"):
            for file in files:
                if file.endswith("_spt.fits"):
                    pre,suf = file.split("_")
                    if pre.endswith("1"):
                        print("removing: ", file)
                        files.remove(file)
                        reduced_list.remove(file)
                    elif pre.endswith("0"):
                        print("removing: ", file)
                        files.remove(file)
                        reduced_list.remove(file)
    
    for k, v in fits_files.items():
        for file in v:
            if file in reduced_list:
                reduced_list.remove(file)
            else:
                continue
    
    singletons = sorted(reduced_list)
    sing_keys = []
    for s in singletons:
        sing_keys.append(s[:9])
    for key in set(sing_keys):
        files = list(glob.glob(f"{key}*.fits"))
        if key in list(fits_files.keys()):
            continue
        else:
            fits_files[key] = files

    return fits_files




def condense_columns(data):
    """
    combines counts for all images (.png, .jpg), fits files (.fits) 
    and a separate combined count for .raw and .flt files
    """
    cols = list(data.columns)
    rf = []
    img = []
    fits = []
    for col in cols:
        if col.split('.')[-1] == 'jpg':
            img.append(col)
        elif col.split('.')[-1] == 'png':
            img.append(col)
        elif col.split('.')[-1] == 'fits':
            temp = col
            fits.append(col)
            if temp.split('.')[0] == 'flt':
                rf.append(col)
            elif temp.split('.')[0] == 'raw':
                rf.append(col)
    df = data.copy()
    # combine raw and flt columns
    rawflt = 0.0
    for colname in rf:
        rawflt += data[colname].values
    df['raw_flt'] = rawflt
    df['raw_flt'] = df['raw_flt'].astype('int64')
    # count imgs
    imgs = 0
    for colname in img:
        imgs += data[colname].values
    df['img'] = imgs
    # count fits
    fitsall = 0
    for colname in fits:
        fitsall += data[colname].values
    df['fits'] = fitsall

    ## Drop columns that were combined above
    df.drop(fits, axis=1, inplace=True)
    df.drop(img, axis=1, inplace=True)
    return df

def input_columns(data, sfx_list):
    """
    collects metadata for input .fits files
    """
    cols = list(data.columns)
    fits = []
    drop = []
    for col in cols:
        if col.split('.')[-1] == 'fits':
            temp = col
            if temp.split('.')[0] in sfx_list:
                fits.append(col)
            else:
                drop.append(col)

    df = data.copy()
    # combine raw and flt columns
    fitscnt = 0.0
    for colname in fits:
        fitscnt += data[colname].values
    df['inputs'] = fitscnt
    df['inputs'] = df['inputs'].astype('int64')

    ## Drop columns that were combined above
    df.drop(fits, axis=1, inplace=True)
    df.drop(drop, axis=1, inplace=True)
    return df


def get_instr(data):
    """
    flag each ipst by instrument
    add instrument column to dataframe
    """
    df = data.copy()
    instr = []
    for i in list(df.index):
        if i[0] == "j":
            instr.append("acs")
        elif i[0] == "o":
            instr.append("stis")
        elif i[0] == "i":
            instr.append("wfc3")
        elif i[0] == "l":
            instr.append("cos")
        else:
            print("error")
    df["instr"] = instr
    return df

def hubble_scatter(df, X, Y='memory', instruments=None, bestfit=False):
    fig = plt.figure(figsize=(11,7))
    ax = fig.gca()

    if instruments is None:
        ax.scatter(df[X], df[Y])
        ax.set_xlabel(X)
        ax.set_ylabel(Y)
        ax.set_title(f"{X} vs. {Y}")
    else:
        cols = list(df.columns)
        if 'instr' in cols:
            instr_col = 'instr'

        else:
            instr_col = 'instr_enc'

        for i in instruments:
            if i == 'acs':
                e = 0
                c = 'blue'
            elif i == 'cos':
                e = 1
                c='lime'
            elif i == 'stis':
                e = 2
                c='red'
            elif i == 'wfc3':
                e = 3
                c = 'orange'
            if instr_col == 'instr':
                ax.scatter((df[X].loc[df[instr_col] == i]), (df[Y].loc[df[instr_col] == i]), c=c, alpha=0.7)
            else:
                ax.scatter((df[X].loc[df[instr_col] == e]), (df[Y].loc[df[instr_col] == e]), c=c, alpha=0.7)
 
            ax.set_xlabel(X)
            ax.set_ylabel(Y)
            ax.set_title(f"{X} vs. {Y}: {[i for i in instruments]}")
            if len(instruments) > 1:
                ax.legend([i for i in instruments])
            
    if bestfit is True:
        x = df[X]
        y = df[Y]
        m, b = np.polyfit(x, y, 1) #slope, intercept
        plt.plot(x, m*x + b, 'k--'); # best fit line
    else:
        plt.show();


# Checking multicollinearity with a heatmap
def multiplot(df, figsize=(20,20), color=None):
    corr = np.abs(df.corr().round(3)) 
    fig, ax = plt.subplots(figsize=figsize)
    if color is None:
        color="Blues"
    mask = np.zeros_like(corr, dtype=np.bool)
    idx = np.triu_indices_from(mask)
    mask[idx] = True
    #xticks = ['fits', 'img', 'raw','mem', 'sec', 'mb','cnt']
    sns.heatmap(np.abs(corr),
                square=True, mask=mask, annot=True, cmap=color, ax=ax)#,
                #xticklabels=xticks, yticklabels=xticks)
    ax.set_ylim(len(corr), -.5, .5)
    return fig, ax

def boxplots(df, x, name=None):
    # iterate over categorical vars to build boxplots of distributions
    # and visualize outliers
    y = 'memory'
    if name is None:
        name = ''
    plt.style.use('seaborn')
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(11,11))
    
    # Create keywords for .set_xticklabels()
    tick_kwds = dict(horizontalalignment='right', 
                      fontweight='light', 
                      fontsize='x-large',   
                      rotation=45)
    sns.boxplot(data=df, x=x, y=y, ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(),**tick_kwds)
    axes[0].set_xlabel(x)
    axes[0].set_ylabel('RAM (mb)')
    # Boxplot with outliers
    axes[0].set_title(f'{name} {x} vs {y}: Boxplot with Outliers')
    
    sns.boxplot(data=df, x=x, y=y, ax=axes[1], showfliers=False)
    axes[1].set_xticklabels(axes[1].get_xticklabels(),**tick_kwds)
    axes[1].set_xlabel(x)
    axes[1].set_ylabel('RAM (mb)')
    axes[1].set_title(f'{name} {x} vs {y}: Outliers Removed')
    fig.tight_layout()
    

    
def percentile_means(df, features, target):
    
    means = {}

    for v in features:
        vals = df[v]

        q25 = vals.quantile(.25)
        q50 = vals.quantile(.5)
        q75 = vals.quantile(.75)
        q95 = vals.quantile(.95)

        m0 = df[target].loc[vals <= q25].mean()
        m25 = df[target].loc[(vals >= q25) & (vals < q50)].mean()
        m50 = df[target].loc[(vals >= q50) & (vals < q75)].mean()
        m75 = df[target].loc[(vals >= q75) & (vals < q95)].mean()
        m95 = df[target].loc[vals >= q95].mean()

        means[v] = [m0, m25, m50, m75, m95]

    # line plt
    plt.figure(figsize=(15, 10))
    for m in means.values():
        plt.plot(m)
    plt.legend([f for f in means.keys()])
    xticks = ['min', '25%', '50%', '75%', '95%']
    plt.xticks(ticks=np.arange(5), labels=xticks, rotation=45)

    return means



    
    

# Detect actual number of outliers for our predictors
def remove_outliers(df):

    out_vars = ['raw_flt', 'img', 'n_files', 'total_mb', 'memory']
    df_outs = df[out_vars]

    # Get IQR scores
    Q1 = df_outs.quantile(0.25)
    Q3 = df_outs.quantile(0.75)
    IQR = Q3 - Q1
    
    # True indicates outliers present
    outliers = (df_outs < (Q1 - 1.5 * IQR)) |(df_outs > (Q3 + 1.5 * IQR))

    # Remove outliers 
    df_zero_outs = df_outs[~((df_outs < (Q1 - 1.5 * IQR)) |(df_outs > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # number of outliers removed
    outs_removed = df_outs.shape[0] - df_zero_outs.shape[0] # 1014
    print(f"outliers removed: ", outs_removed)
    print(f"orig df shape: ", df.shape)
    print(f"new df shape: ", df_zero_outs.shape)
    return df_zero


# Choose a linear model by forward selection
# The function below optimizes adjusted R-squared by adding features that help the most one at a time
# until the score goes down or you run out of features.


def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    Example:
    model = forward_selected(data, 'sl')
    print(model.model.formula)
    # sl ~ rk + yr + 1
    print(model.rsquared_adj)
    # 0.835190760538
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    
    print(model.model.formula)
    print(model.rsquared_adj)
    print(model.summary())
    
    return model


def kfold_ols(df_fin, k=10):
    # Run k-fold cross validation
    y = df_fin['memory']
    X = df_fin.drop('memory', axis=1)
    results = [['set#','R_square_train','MSE_train','R_square_test','MSE_test']]
    num_coeff = X.shape[1]
    list_predictors = [str(x) for x in X.columns]
    list_predictors.append('intercept') 
    reg_params = [list_predictors]

    i=0
    k=k
    while i <(k+1):
        X_train, X_test, y_train, y_test = train_test_split(X,y)
        data = pd.concat([X_train,y_train], axis=1)
        f = 'memory~C(raw_flt)+C(img)+n_files+total_mb' 
        model = smf.ols(formula=f, data=data).fit()
        model.summary()

        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)

        train_residuals = y_hat_train - y_train
        test_residuals = y_hat_test - y_test

        train_mse = metrics.mean_squared_error(y_train, y_hat_train)
        test_mse = metrics.mean_squared_error(y_test, y_hat_test)

        R2_train = metrics.r2_score(y_train,y_hat_train)
        R2_test = metrics.r2_score(y_test,y_hat_test)

        results.append([i, R2_train, train_mse, R2_test, test_mse])
        i+=1

    results = pd.DataFrame(results[1:],columns=results[0])
    return model, results





# # Tukey's method using IQR to eliminate 
# def detect_outliers(df, n, features):
#     outlier_indices = []
#     # iterate over features(columns)
#     for col in features:
#         # 1st quartile (25%)
#         Q1 = np.percentile(df[col], 25)
#         # 3rd quartile (75%)
#         Q3 = np.percentile(df[col],75)
#         # Interquartile range (IQR)
#         IQR = Q3 - Q1
#         # outlier step
#         outlier_step = 1.5 * IQR
#         # Determine a list of indices of outliers for feature col
#         outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
#         # append the found outlier indices for col to the list of outlier indices 
#         outlier_indices.extend(outlier_list_col)
#         # select observations containing more than 2 outliers
#         outlier_indices = Counter(outlier_indices)        
#         multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
#         return multiple_outliers 
# # Outliers_to_drop = detect_outliers(data,2,["col1","col2"])
# # df.loc[Outliers_to_drop] # Show the outliers rows
# # Drop outliers
# # data= data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)






def build_classifier(input_shape=None, layers=None, input_name="hst_jobs", 
                     output_name="memory_pred"):
    if input_shape is None:
        input_shape = 9
    if layers is None:
        layers = [18, 32, 64, 32, 18, 9]
    model = Sequential()
    # visible layer
    inputs = keras.Input(shape=(input_shape,), name=input_name)
    # hidden layers
    x = keras.layers.Dense(layers[0], activation='relu', name='dense_1')(inputs)
    for i, layer in enumerate(layers[1:]):
        i+=1
        x = keras.layers.Dense(layer, activation="relu", name=f"dense_{i+1}")(x)
    # output layer
    outputs = keras.layers.Dense(4, activation="softmax", name=output_name)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="sequential_mlp")
    model.compile(loss="categorical_crossentropy",
                    optimizer='adam', metrics=["accuracy"])
    return model


def build_regressor(input_shape=None, layers=None, input_name="hst_jobs", 
                    output_name="wallclock_reg"):
    if input_shape is None:
        input_shape = 9
    if layers is None:
        layers = [18, 32, 64, 32, 18, 9]
    model = Sequential()
    # visible layer
    inputs = keras.Input(shape=(input_shape,), name=input_name)
    # hidden layers
    x = keras.layers.Dense(layers[0], activation='relu', name='dense_1')(inputs)
    for i, layer in enumerate(layers[1:]):
        i+=1
        x = keras.layers.Dense(layer, activation="relu", name=f"dense_{i+1}")(x)
    # output layer
    outputs = keras.layers.Dense(1, name=output_name)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="sequential_mlp")
    model.compile(loss="mean_squared_error",
                    optimizer='adam', metrics=["accuracy"])
    return model

# def build_model(input_shape=None, layers=None, input_name="jobs", output_name="memory_pred"):
#     if input_shape is None:
#         input_shape = 11
#     if layers is None:
#         layers = [11, 9, 7, 9, 11]
#     model = Sequential()
#     # visible layer
#     inputs = keras.Input(shape=(input_shape,), name=input_name)
#     # hidden layers
#     x = keras.layers.Dense(layers[0], activation='relu', name='dense_1')(inputs)
#     for i, layer in enumerate(layers[1:]):
#         i+=1
#         x = keras.layers.Dense(layer, activation="relu", name=f"dense_{i+1}")(x)
#     # output layer
#     outputs = keras.layers.Dense(1, name=output_name)(x)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="sequential_mlp")
#     model.compile(loss="mean_squared_error",
#                     optimizer='adam', metrics=["accuracy"])
#     return model

def fit(model, X_train, y_train, X_test, y_test, 
        verbose=1, epochs=200, batch_size=10, callbacks=None):
    validation_data = (X_test, y_test)
    start = dt.datetime.now(tz=tz.get_localzone()).strftime('%m/%d/%Y - %I:%M:%S %p')
    print("TRAINING STARTED: ", start)
    history = model.fit(X_train, y_train, batch_size=batch_size, 
                        validation_data=validation_data, verbose=verbose, 
                        epochs=epochs, callbacks=callbacks)
    end = dt.datetime.now(tz=tz.get_localzone()).strftime('%m/%d/%Y - %I:%M:%S %p')
    print("TRAINING COMPLETE: ", end)
    model.summary()
    return history


def get_scores(model, X_train, X_test, y_train, y_test, verbose=False):
    train_scores = model.evaluate(X_train, y_train, verbose=2)
    test_scores = model.evaluate(X_test, y_test, verbose=2)
    if verbose:
        print("Train accuracy:", np.round(train_scores[1],2))
        print("Test accuracy:", np.round(test_scores[1],2))
    print("\nTrain loss:", np.round(train_scores[0],2))
    print("\nTest loss:", np.round(test_scores[0],2))

    return test_scores


def predict_clf(model, X_train, X_test, y_train, y_test):
    predictions = np.argmax(model.predict(X_test), axis=-1) +1
    print("\nPredictions:")
    return predictions

def predict_reg(model, X_train, X_test, y_train, y_test):
    from sklearn.metrics import mean_squared_error as MSE
    # predict results of training set
    y_hat = model.predict(X_train)
    rmse_train = np.sqrt(MSE(y_train, y_hat)) 
    print("RMSE Train : % f" %(rmse_train))
    # # predict results of test set
    y_pred = model.predict(X_test)
    # RMSE Computation 
    rmse_test = np.sqrt(MSE(y_test, y_pred)) 
    print("RMSE Test : % f" %(rmse_test))
    #train_preds = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
    np.set_printoptions(precision=2)
    preds = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
    return preds

def get_resid(preds):
    res = []
    for p, a in preds:
        # predicted - actual
        res.append(p-a)
    print(max(res))
    print(min(res))
    print(np.mean(res))

    fig1 = plt.figure(figsize=(11,7))
    ax1 = fig1.gca()
    ax1 = sns.regplot(x=res, y=preds[:, 0], data=None, scatter=True, color='red')
    plt.show()
    fig2 = plt.figure(figsize=(11,7))
    ax2 = fig2.gca()
    ax2 = sns.regplot(x=res, y=preds[:, 0], fit_reg=True)
    plt.show()
    return res

def get_preds(X,y,model=None,verbose=False):
    if model is None:
        model=model
    # class predictions 
    #y_true = y.flatten()
    y_true = np.argmax(y, axis=-1) + 1
    y_pred = np.argmax(model.predict(X), axis=-1) +1
    #y_pred = model.predict_classes(X).flatten() 
    preds = pd.Series(y_pred).value_counts(normalize=False)
    
    if verbose:
        print(f"y_pred:\n {preds}")
        print("\n")

    return y_true, y_pred


def get_resid(preds):
    res = []
    for p, a in preds:
        # predicted - actual
        res.append(p-a)
    print(max(res))
    print(min(res))
    print(np.mean(res))

    fig1 = plt.figure(figsize=(11,7))
    ax1 = fig1.gca()
    ax1 = sns.regplot(x=res, y=preds[:, 0], data=None, scatter=True, color='red')
    plt.show()
    fig2 = plt.figure(figsize=(11,7))
    ax2 = fig2.gca()
    ax2 = sns.regplot(x=res, y=preds[:, 0], fit_reg=True)
    plt.show()
    return res

def keras_history(history, figsize=(10,4)):
    """
    side by side sublots of training val accuracy and loss (left and right respectively)
    """
    
    import matplotlib.pyplot as plt
    
    fig,axes=plt.subplots(ncols=2,figsize=(15,6))
    axes = axes.flatten()

    ax = axes[0]
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')

    ax = axes[1]
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def save_model(model, version="0001", name='Sequential', save=True, weights=True):
    '''The model architecture, and training configuration (including the optimizer, losses, and metrics) 
    are stored in saved_model.pb. The weights are saved in the variables/ directory.'''
    model_subfolder = './models'
    model_version = version
    model_name = name
    path = os.path.join(model_subfolder, model_name)
    model_path = os.path.join(path, model_version)
    weights_path = f"{model_subfolder}/{model_name}/weights/ckpt_{model_version}"
    if save is True:
        model.save(model_path)
    if weights is True:
        model.save_weights(weights_path)
        #model.save_weights(weights)
    for root, dirs, files in os.walk(path):
            indent = '    ' * root.count(os.sep)
            print('{}{}/'.format(indent, os.path.basename(root)))
            for filename in files:
                print('{}{}'.format(indent + '    ', filename))
    return model_path

def fusion_matrix(matrix, classes=None, normalize=True, title='Fusion Matrix', cmap='Blues',
    print_raw=False): 
    """
    FUSION MATRIX
    -------------    
    matrix: can pass in matrix or a tuple (ytrue,ypred) to create on the fly 
    classes: class names for target variables
    """


    from sklearn import metrics                       
    from sklearn.metrics import confusion_matrix 
    import itertools
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    # make matrix if tuple passed to matrix:
    if isinstance(matrix, tuple):
        y_true = matrix[0].copy()
        y_pred = matrix[1].copy()
        
        if y_true.ndim>1:
            y_true = y_true.argmax(axis=1)
        if y_pred.ndim>1:
            y_pred = y_pred.argmax(axis=1)
        fusion = metrics.confusion_matrix(y_true, y_pred)
    else:
        fusion = matrix
    
    # INTEGER LABELS
    if classes is None:
        classes=set(matrix[0])
        #classes=list(range(len(matrix)))

    #NORMALIZING
    # Check if normalize is set to True
    # If so, normalize the raw fusion matrix before visualizing
    if normalize:
        fusion = fusion.astype('float') / fusion.sum(axis=1)[:, np.newaxis]
        thresh = 0.5
        fmt='.2f'
    else:
        fmt='d'
        thresh = fusion.max() / 2.
    
    # PLOT
    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(fusion, cmap=cmap, aspect='equal')
    
    # Add title and axis labels 
    plt.title(title) 
    plt.ylabel('TRUE') 
    plt.xlabel('PRED')
    
    # Add appropriate axis scales
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    #ax.set_ylim(len(fusion), -.5,.5) ## <-- This was messing up the plots!
    
    # Text formatting
    fmt = '.2f' if normalize else 'd'
    # Add labels to each cell
    #thresh = fusion.max() / 2.
    # iterate thru matrix and append labels  
    for i, j in itertools.product(range(fusion.shape[0]), range(fusion.shape[1])):
        plt.text(j, i, format(fusion[i, j], fmt),
                horizontalalignment='center',
                color='white' if fusion[i, j] > thresh else 'black',
                size=14, weight='bold')
    
    # Add a legend
    plt.colorbar()
    plt.show() 
    return fusion, fig

