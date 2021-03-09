# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:35:54 2021

@author: lloth
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from pca_mod import shift_pca
from pca_mod import shift

#%%
df = pd.read_csv('f9ad.csv')


def tape(data,
                index,
                target,
                split = 1,
                drops = [],
                hGRU = 386,
                   hDrop1 = 0,
                   hDrop2 = 0, 
                   hDrop3 = 0,
                   hDense1 = 1,
                   hDense2 = 32,
                   hDense3 = 128,
                   hDense4 = 1,
                   hDense5 = 128,
                   asel_choice = 'pca',
                   hMemoryMeters = 25,
                   hstep_extension = 5,
                   hAttrCount = 3,
                   imagination_meters = 25,
                   verbose=0,
                   future = 0.15,
                   plot_samples = True,
                   convert_to_diff = [],
                   lcs_list = [],
                   clean=False,
                   shift = 0.1,
                   resample='radius',
                   resample_coef = 1,
                   resample_weights='distance'):


    df = data
    
    ## List attributes
    # print('Listing all the attributes in the loaded dataset:')
    # print(list(df))
    
    #%%
    
    ## Drop unwanted columns
    
    
    
    df = df.drop(columns = drops)
    #df = df[1900:]   ## Done this to include more data. Smart selection needed!
    #%%
    
    ## Select target attribute and index.
    
    #index = 'Measured Depth m'
    #target =  'MWD Continuous Inclination dega'
    
    
    #%%
    
    ## Get basic statistics extraction
    #
    # s = global statistics
    # m = matrix of gap lengths
    # per = statistics per attribute
    
    
    if clean == False:
        from statistics_module import stats
        
        s, m, per = stats(df)
        
        #%%
        
        ## Gap statistics for target
        #
        # This chart will show the percentage of dataset occupied by gaps of a certain
        # size. Gaps are normal in drilling logs and nothing to be afraid of
        
        x_label = per[target]['gap_sizes']
        x = np.arange(0, len(x_label),1)
        
        y = per[target]['percentage_cells_occupied'] 
        # plt.xticks(x, x_label, rotation=90)
        # plt.bar(x,y)
        # plt.title(f'Gap distribution in:\n {target}')
        # plt.xlabel('Gap length')
        # plt.ylabel('Percentage of dataset occupied')
        
        x_labels = x.tolist()
        x_labels[0] = 'data'
        # plt.xticks(x, x_labels)
        # plt.grid()
        # plt.show() 
        
        #%%
        
        ## Outlier detection
        
        outlier_cutoff = 0.005 #arbitrarily selected
        
        # calculation that penalizes long, rare, continuous gaps
        out_coef = per[target]['gap_sizes'] / (per[target]['gap_counts'] * len(df))
        
        x = np.arange(0,len(per[target]['gap_sizes']),1)
        x_label = per[target]['gap_sizes']
        x = np.arange(0, len(x_label),1)
        
        # plt.xticks(x, x_label, rotation=90)
        # plt.bar(x,out_coef)
        # #plt.ylim(0,0.005)
        # plt.plot(x,[outlier_cutoff]*len(x), color='red', label='cutoff')
        # plt.legend()
        x_labels = x.tolist()
        x_labels[0] = 'data'
        # plt.xticks(x, x_labels)
        # plt.title(f'Gap occupancy = gap size / (relative gap quantity) \n in:{target}')
        # plt.xlabel('Gap length')
        # plt.ylabel('Gap occupancy')
        # plt.grid()
        # plt.show() 
        
        #%%
        
        ## Automatic proposal of useful area, part 1
        
        # find the smallest outlier gap
        # outlier coefficients - True when above outlier cutoff
        cutoff_map = (out_coef >= outlier_cutoff)
        
        # Using map created above, what is the smallest outlier?
        lower_cutoff = np.min(np.asarray(per[target]['gap_sizes'])[cutoff_map])
        
        # This is done to quickly mark gaps bigger than cutoff with zero and
        # other with NaN. This makes a good chart.
        from functools import partial
        
        def _cutoff(x, lower_cutoff=0):
            if x >= lower_cutoff:
                return 0
            else:
                return np.nan
        
        _cutoff_par = partial(_cutoff, lower_cutoff=lower_cutoff)
        mapped_outliers = list(map(_cutoff_par, m[target]))
        
        # plt.scatter(df[index], df[target], s=1, c='black', label='data')
        # plt.plot(df[index], mapped_outliers, c='red', label='Unusuable range') 
        #                                                 # has to be plot to avoid index
        #                                                 # discontinuities
                                                        
        # plt.grid()
        # plt.legend()
        # plt.title('Useful range analysis')
        # plt.xlabel(f'{index}')
        # plt.ylabel(f'{target}')
        # plt.show()
        #%%
        
        ## Automatic proposal of useful area, part 2
        
        # Simply finds the biggest area with acceptable gaps
        
        # TODO - check if the algorithm will detect a stride at the end of the dataset
        #        because I have a feeling it won't!
        
        strides = []
        
        s_start = -1
        s_stop = -1
        
        for i in range(len(df)):
            if mapped_outliers[i] != 0 and s_start == -1:
                s_start = i
            elif mapped_outliers[i] == 0 and s_start != -1:
                s_stop = i
                strides.append([s_start, s_stop, s_stop - s_start ])
                
                s_start = -1
                s_stop = -1
        
        strides = np.asarray(strides)
        strides = strides[strides[:,2].argsort()][::-1] # sort by length [2] 
                                                        # and reverse
        # print(f'''Proposed range to use is row {strides[0,0]} to row {strides[0,1]}
        # for total of {strides[0,0]} rows
        #       ''')
        # print(f'All found strides are: [start, stop, length]')
        # print(strides)
        
        s_start = strides[0,0]
        s_stop = strides[0,1]
        #%%
        
        ## cut the dataframe for the selected stride, redo the stats
        #  From now on dfs is used (dataframe stride)
        margin_percent = 1  # since the edges can be a bit unpredictable, margin is
                            # removed
        
        s_start = s_start + int((s_stop - s_start) * 0.01*margin_percent)
        s_stop = s_stop - int((s_stop - s_start) * 0.01*margin_percent)
        dfs = df.iloc[s_start:s_stop] # dfs = DataFrameStride
        s, m, per = stats(dfs)
        
        #%%
        
        ## Removing columns that contain big gaps
        
        from functools import partial
        
        def _cutoff_inv(x, lower_cutoff=0):
            if x >= lower_cutoff:
                return 1
            else:
                return 0
        
        _cutoff_par = partial(_cutoff_inv, lower_cutoff=lower_cutoff)
        
        killed_cols = []
        
        for column in list(dfs):
            mapped_outliers = list(map(_cutoff_par, m[column]))
            offender_count = np.sum(mapped_outliers)
            if offender_count > 0:
                dfs = dfs.drop(columns = [column])
                killed_cols.append([column, 100*offender_count/len(dfs)])
        
        killed_cols = pd.DataFrame(killed_cols, columns=['Name','Percent offending'])
        #print('Removed following columns due to outlier gap (showing under 15% only):')
        #print(killed_cols.sort_values(by='Percent offending')[killed_cols['Percent offending'] < 15])
        
        #%%
        
        ## Checking if first derivative of index is stable.
        
        index_dr = np.diff(dfs[index])
        
        index_mean = np.mean(index_dr)
        index_std = np.std(index_dr)
        index_maxgap = np.max(index_dr)
        deviation = np.abs(index_dr - index_mean)/index_std
        
        #print(f'Maximum distance from mean is {np.max(deviation):.1f} standard deviations')
        #print(f'If this value is above 6, there may be too high sampling frequency variation')
        
        #%%
        
        ## Counting zeros in the first derivative to see if it should be ffilled
        ## or linearly interpolated
        
        ## NOTE: Actual filling will not happen here, but AFTER the data split
        
        fill_method = {}
        
        for attribute in list(dfs):
            
            dropna_diff = np.diff(dfs[attribute].dropna())
            zeros_p = np.count_nonzero(dropna_diff == 0) / len(dropna_diff)
            
            if zeros_p > 0.9: # Threshold to check?
                fill_method[attribute] = 'ffill'
            else:
                fill_method[attribute] = 'linterp'
        
        
        #%%
        
        #%% 
        
        ## Gap filling - but only forward filling. Linear interpolation is done later
        
        for attribute in list(dfs):
            if fill_method[attribute] == 'ffill':
                dfs[attribute] = dfs[attribute].ffill().rolling(5, center=True).mean().ffill().bfill()
        
    #%%
    
    else:
        dfs = df.copy()
    
    #%%
    
    ## DATA SPLIT
    
    #split = 0.6 #portion of data available
    #future = 0.15 #section after available, for testing
    
    X = dfs.drop(target, axis=1)
    y = dfs[target].to_frame()
    
    
    if split != 1:
        splitpoint = int(len(dfs)*split)
        futurepoint = int(len(dfs)*(split+future))
        
        X_train = X[:splitpoint]
        y_train = y[:splitpoint]
        X_test  = X[splitpoint:futurepoint]
        y_test  = y[splitpoint:futurepoint]
    else:
        X_train = X
        y_train = y
        X_test  = X
        y_test  = y
    
    
    #%%
    
    ## Linear interpolation after split, so the future does not leak.
    
    if clean == False:
        for attribute in list(X_train):
            if fill_method[attribute] == 'linterp':
                X_train = X_train.interpolate().ffill().bfill()#.rolling(5, center=True).mean().ffill().bfill()
                
        for attribute in list(y_train):
            if fill_method[attribute] == 'linterp':
                y_train = y_train.interpolate().ffill().bfill()#.rolling(5, center=True).mean().ffill().bfill()
                
        for attribute in list(X_test):
            if fill_method[attribute] == 'linterp':
                X_test = X_test.interpolate().ffill().bfill()#.rolling(5, center=True).mean().ffill().bfill()
                
        for attribute in list(y_test):
            if fill_method[attribute] == 'linterp':
                y_test = y_test.interpolate().ffill().bfill()#.rolling(5, center=True).mean().ffill().bfill()
    
    
    
    
    
    
    
    #%%
    
    ## Resampling
    if clean == False:
        

        
        
        step_length = index_mean * hstep_extension
        
        i_train_min = np.min(X_train[index])
        i_train_max = np.max(X_train[index])
        i_test_min = np.min(X_test[index])
        i_test_max = np.max(X_test[index])
        
        index_train = np.arange(i_train_min, i_train_max, step_length).reshape(-1,1)
        index_test = np.arange(i_test_min, i_test_max, step_length).reshape(-1,1)
        
        if resample == 'radius':
            from sklearn.neighbors import RadiusNeighborsRegressor
            reg = RadiusNeighborsRegressor(radius=index_maxgap*resample_coef, weights=resample_weights)
        elif resample == 'knn':
            from sklearn.neighbors import KNeighborsRegressor
            reg = KNeighborsRegressor(weights=resample_weights, n_neighbors=resample_coef)
        else:
            sys.exit("Error, incorrect resampling algorithms choice")
        
        reg.fit(X_train[index].to_numpy().reshape(-1,1), y_train[target].to_numpy())
        y_train = pd.DataFrame()
        y_train[target] = reg.predict(index_train)
        
        reg.fit(X_test[index].to_numpy().reshape(-1,1), y_test[target].to_numpy())
        y_test = pd.DataFrame()
        y_test[target] = reg.predict(index_test)
        
        X_train_resampled = pd.DataFrame()
        for attribute in list(X_train):
            reg.fit(X_train[index].to_numpy().reshape(-1,1), X_train[attribute].to_numpy())
            X_train_resampled[attribute] = reg.predict(index_train)
            
        X_train = X_train_resampled
        
        
        
        X_test_resampled = pd.DataFrame()
        for attribute in list(X_train):
            reg.fit(X_test[index].to_numpy().reshape(-1,1), X_test[attribute].to_numpy())
            X_test_resampled[attribute] = reg.predict(index_test)
            
        X_test = X_test_resampled
    #%%
    
    ## Inclination to delta inclination convertion needed here!
    
    #list of parameters that are to be in local coordinate system
    
    for attr in convert_to_diff:
        if attr == target:
            y_train[attr] = y_train[attr].diff().bfill() #bfill to kill initial NaN
            y_test[attr] = y_test[attr].diff().bfill()
        else:
            X_train[attr] = X_train[attr].diff().bfill()
            X_test[attr] = X_test[attr].diff().bfill()
    #%%
    
    ## Scaling the data. Note that range is decide on the training dataset only!
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    pca_allattr = X_train.columns
    X_train[X_train.columns] = scaler_X.fit_transform(X_train[X_train.columns])
    y_train[y_train.columns] = scaler_y.fit_transform(y_train[y_train.columns])
    
    ## Test portion is tranformed based on the existing scaler
    X_test[X_test.columns] = scaler_X.transform(X_test[X_test.columns])
    y_test[y_test.columns] = scaler_y.transform(y_test[y_test.columns])
    
    
    #%%
    
    ## Dataframe for use in correlation analysis, where X_train and y_train is
    ## together
    df_train = X_train
    df_train = df_train.merge(y_train, how='outer', left_index=True,
                              right_index=True)
    
    #%%
    
    ## Choice of attribute selection method done on the complete dataset
    ## NOTE: PCA has to be applied AFTER the split, never before!
    
    # print('''Choose attribute selection method:
    #     1) pearson coefficient
    #     2) PCA
    #     3) ppscore''')
        
    #asel_choice = 'pca'#input('Your choice:')
    
    #%%
    ## Simple correlation, pearson, straight from Pandas
    
    ## Note that correlations are re-done after each split, and done only on the
    ## training dataset!
    
    ## [] Ensure that Index is carried forward!
    
    PCA_n = -1 #i.e. not in use
    
    
    if asel_choice == 'pearson':
        

        dfs_corr = df_train.corr(method='pearson')
        corr_values = dfs_corr[target].to_numpy()
        corr_index = dfs_corr[target].index.to_numpy()

        corr_m = np.column_stack((corr_values, corr_index))

        for i in range(len(corr_m)):
            if np.isnan(corr_m[i,0]):
                corr_m[i,0] = 0
            else:
                corr_m[i,0] = np.abs(corr_m[i,0])

        corr_m = corr_m[corr_m[:,0].argsort()]

        keep_columns = corr_m[-1-hAttrCount:-1,1]

        X_train = X_train[keep_columns]
        X_test  = X_test[keep_columns]
    
    
    
    ## PCA based 
    
    
    elif asel_choice == 'pca':
        from sklearn.decomposition import PCA
        
        keep_columns = [] #empty for future code compatibility

            
        PCA_n = hAttrCount
        
        # applied after sensitivity analysis
    
    ## ppscore based
    
    elif asel_choice == 'ppscore':
        import ppscore as pps
        dfs_corr = pps.predictors(df_train, target, output='list')
        
    
        corr_values = []
        corr_index = []
        for i in dfs_corr:
            corr_values.append(i['ppscore'])
            corr_index.append(i['x'])
            
            
        corr_m = np.column_stack((corr_values, corr_index))
        
        
        corr_m = corr_m[corr_m[:,0].argsort()]
        
        keep_columns = corr_m[-1-hAttrCount:-1,1]
        
        X_train = X_train[keep_columns]
        X_test  = X_test[keep_columns]
        
    else:
        sys.exit("Error, incorrect attribute selection choice")
    


    #%%
    
    ## Applying PCA. It is done late to preserve all attributes for sensitivity test
    if PCA_n != -1:
        scaler_pca = MinMaxScaler() # new scaler here because PCA can push 
                                    # variables out of (-1,1) bounds
                                    
        pca = PCA(n_components = PCA_n)
        
        X_train = pca.fit_transform(X_train)
        X_train = scaler_pca.fit_transform(X_train)
        
        X_test = pca.transform(X_test)
        X_test = scaler_pca.transform(X_test)
    
    
    
    
    
    #%%
    
    ## Data shaping
    
    ## from now on, arrays are being morphed into shapes valid for RNN+MLP
    
    if clean == True:
        step_length = 1
    memory = int(hMemoryMeters/step_length)
    
    imagination = int(imagination_meters/step_length)
    
    X_attr = list(X_train)
    
    try:
        X_train = X_train.to_numpy()    
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()    
    except:
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
    
    if split != 1:
        X_test = np.concatenate([X_train[-memory+1:,:], X_test], axis=0)
        y_test = np.concatenate([y_train[-memory+1:], y_test], axis=0)
    #%%
    
    def prepare(data, start, stop, cut_margin = 0, lcs=False):
        memory = stop-start
        stack = []
        for i in range(memory):
            stack.append(np.roll(data, -i))
    
        stack = np.flip(np.rot90(stack), axis=0)[start:-memory+1-cut_margin]
    
        if lcs == True:
            zero = stack[:,0]
    
            for j in range(len(zero)):
                stack[j] = stack[j] - zero[j]
        return stack
    
    target_lcs_correction = 1
    
    if target in lcs_list:
        X_train_RNN = prepare(np.squeeze(y_train), 0, memory, cut_margin = imagination, lcs=True)
        X_test_RNN = prepare(np.squeeze(y_test), 0, memory, cut_margin = imagination, lcs=True)
        
        y_train_RNN = prepare(np.squeeze(y_train), memory, memory+imagination, lcs=True)
        y_test_RNN = prepare(np.squeeze(y_test), memory, memory+imagination, lcs=True)
        
        offset_train = X_train_RNN[:,-1]
        offset_test = X_test_RNN[:,-1]
        
        for k in range(len(offset_train)):
            y_train_RNN[k] = y_train_RNN[k] + offset_train[k]
            
        for k in range(len(offset_test)):
            y_test_RNN[k] = y_test_RNN[k] + offset_test[k]
        
        target_lcs_correction = 1/np.max(y_train_RNN)
        
        y_train_RNN = y_train_RNN * target_lcs_correction
        X_train_RNN = X_train_RNN  * target_lcs_correction
        y_test_RNN = y_test_RNN * target_lcs_correction
        X_test_RNN = X_test_RNN * target_lcs_correction
        
    else:
        X_train_RNN = prepare(np.squeeze(y_train), 0, memory, cut_margin = imagination)
        X_test_RNN = prepare(np.squeeze(y_test), 0, memory, cut_margin = imagination)
        y_train_RNN = prepare(np.squeeze(y_train), memory, memory+imagination)
        y_test_RNN = prepare(np.squeeze(y_test), memory, memory+imagination)
    
    #%%
    
    
    
    
    X_train_MLP = []
    X_test_MLP = []
    
    #%%
    if PCA_n == -1:
        X_lcs_correction = [1]*len(X_train[0])
        
        for i in range(len(X_train[0])):
            if keep_columns[i] in lcs_list:
                X_train_MLP.append(prepare(X_train[:,i],memory,memory+imagination, lcs=True))
                X_lcs_correction[i] = 1/np.max(X_train_MLP[i])
                X_train_MLP[i] = X_train_MLP[i]*X_lcs_correction[i]
            else:
                X_train_MLP.append(prepare(X_train[:,i],memory,memory+imagination))
            
        X_train_MLP = np.asarray(X_train_MLP)
        X_train_MLP = np.concatenate(X_train_MLP[:,:, np.newaxis], axis = 1)
        X_train_MLP = np.rot90(X_train_MLP, axes=(1,2), k=3)
        
        
        
        
        
        for i in range(len(X_test[0])):
            if keep_columns[i] in lcs_list:
                X_test_MLP.append(prepare(X_test[:,i],memory,memory+imagination, lcs=True))
                X_test_MLP[i] = X_test_MLP[i]*X_lcs_correction[i]
            else:
                X_test_MLP.append(prepare(X_test[:,i],memory,memory+imagination))
            
        X_test_MLP = np.asarray(X_test_MLP)
        X_test_MLP = np.concatenate(X_test_MLP[:,:, np.newaxis], axis = 1)
        X_test_MLP = np.rot90(X_test_MLP, axes=(1,2), k=3)
        
    else:
        for i in range(len(X_train[0])):
            X_train_MLP.append(prepare(X_train[:,i],memory,memory+imagination))
                
        X_train_MLP = np.asarray(X_train_MLP)
        X_train_MLP = np.concatenate(X_train_MLP[:,:, np.newaxis], axis = 1)
        X_train_MLP = np.rot90(X_train_MLP, axes=(1,2), k=3)
            
        for i in range(len(X_test[0])):
            X_test_MLP.append(prepare(X_test[:,i],memory,memory+imagination))
                
        X_test_MLP = np.asarray(X_test_MLP)
        X_test_MLP = np.concatenate(X_test_MLP[:,:, np.newaxis], axis = 1)
        X_test_MLP = np.rot90(X_test_MLP, axes=(1,2), k=3)
    #%%
    
    X_train_RNN_m = X_train_RNN[:,:,np.newaxis]
    X_train_m = [X_train_RNN_m, X_train_MLP]
    
    X_test_RNN_m = X_test_RNN[:,:,np.newaxis]
    X_test_m = [X_test_RNN_m, X_test_MLP]
    #%%
        
        
        
    #%%
    
    ## Local coordinate system
    
    ## Just a cumsum on parameter converted to delta earlier
    
    
    #%%
    
    ## ML model definition
    
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import (Dense, Dropout, GRU, Flatten,
                                         GaussianNoise, concatenate, LSTM,
                                         Bidirectional, TimeDistributed)
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import MaxPool1D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.callbacks import ModelCheckpoint
    import tensorflow as tf
    
    from tensorflow.keras.models import load_model
    
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    tf.keras.backend.clear_session()
    
    visible1 = Input(shape=(memory,1))
    
    
    visible2 = Input(shape=((imagination),len(X_train[0])))

    x1 = TimeDistributed(Dense(hDense4))(visible1) 
    x1 = GRU(units=hGRU, kernel_initializer =  'glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer="zeros", kernel_regularizer='l2', recurrent_regularizer=None,
               bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
               recurrent_constraint=None, bias_constraint=None, return_sequences=True,
               return_state=False, stateful=False)(x1)

    x1 = Dense(imagination)(x1)
    x1 = Flatten()(x1)
    x1 = Dropout(hDrop1)(x1)



    x2 = TimeDistributed(Dense(hDense5))(visible2)
    dense2 = Dense(hDense1, activation="linear")(x2)
    drop2 = Dropout(hDrop2)(dense2)
    flat2 = Flatten()(drop2)
    dense2 = Dense(imagination, activation='linear')(flat2)
    drop2 = Dropout(hDrop3)(flat2)
    
    combined = concatenate([x1, drop2])
    
    z = Dense(hDense3, activation="relu")(combined)
    z = Dense(imagination, activation="linear")(z)

    
    model = Model(inputs=[visible1, visible2], outputs=z)
    
     
    
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
    
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss',
                                 mode='min', save_best_only=True, verbose=0)
    

    model.compile(optimizer='adam',loss='mean_squared_error')
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    ## Training    
    rowcount = len(y_train_RNN)
    val_border = int(rowcount*0.85)
    
    X_train_m_a = []
    X_train_m_b = []
    
    X_train_m_a.append(X_train_m[0][:val_border])
    X_train_m_a.append(X_train_m[1][:val_border])
    
    X_train_m_b.append(X_train_m[0][val_border:])
    X_train_m_b.append(X_train_m[1][val_border:])
    
    
    
    y_train_RNN_a = y_train_RNN[:val_border]
    y_train_RNN_b = y_train_RNN[val_border:]
    
    
    history = model.fit(X_train_m_a,y_train_RNN_a,validation_data=(X_train_m_b,y_train_RNN_b),
                                    epochs=2000, verbose=verbose, batch_size=32,
                                    callbacks=[es, mc])
    
    model = load_model('best_model.h5')
    

    result_test = model.evaluate(X_test_m, y_test_RNN, verbose=0)
    
    
    if split == 1:
        singular_sensitivity = []
        if PCA_n != -1:
            for i in range(pca.n_features_):
                X_test_MLP_plus = shift_pca(X_test_MLP, 
                                           scaler_pca, 
                                           pca, 
                                           channel=i, 
                                           shift=shift)
                
                X_test_m_plus = [X_test_RNN_m, X_test_MLP_plus]
                results_plus = scaler_y.inverse_transform(model.predict(X_test_m_plus))
    
                X_test_MLP_minus = shift_pca(X_test_MLP, 
                                           scaler_pca, 
                                           pca, 
                                           channel=i, 
                                           shift=-shift)
                X_test_m_minus = [X_test_RNN_m, X_test_MLP_minus]
                results_minus = scaler_y.inverse_transform(model.predict(X_test_m_minus))
                
                if target in convert_to_diff:
                    results_plus = np.cumsum(results_plus, axis=1)
                    results_minus = np.cumsum(results_minus, axis=1)
                
                sens = (results_plus - results_minus)/2
                
                
                ave = np.average(sens, axis=0)
                perc5 = np.percentile(sens,5,axis=0)
                perc25 = np.percentile(sens,25,axis=0)
                perc50 = np.percentile(sens,50,axis=0)
                perc75 = np.percentile(sens,75,axis=0)
                perc95 = np.percentile(sens,95,axis=0)
                
                
                #plt.plot(ave, linewidth=2, color='darkblue')
                plt.plot(perc5, linewidth=1, linestyle=":", color='black')
                plt.plot(perc25, linewidth=1, color='black')
                plt.plot(perc50, linewidth=2, color='black')
                plt.plot(perc75, linewidth=1, color='black')
                plt.plot(perc95, linewidth=1, linestyle=":", color='black')
                plt.title(pca_allattr[i])
                plt.grid()
                plt.show()
                singular_sensitivity.append(np.average((results_plus - results_minus)/2))
                
            print(singular_sensitivity)
            print(pca_allattr)
        else:
            for i in range(len(keep_columns)):
                X_test_MLP_plus = shift(X_test_MLP,
                                           channel=i, 
                                           shift=shift)
                X_test_m_plus = [X_test_RNN_m, X_test_MLP_plus]
                results_plus = scaler_y.inverse_transform(model.predict(X_test_m_plus))
    
                X_test_MLP_minus = shift(X_test_MLP,
                                           channel=i, 
                                           shift=-shift)
                X_test_m_minus = [X_test_RNN_m, X_test_MLP_minus]
                results_minus = scaler_y.inverse_transform(model.predict(X_test_m_minus))
                
                if target in convert_to_diff:
                    results_plus = np.cumsum(results_plus, axis=1)
                    results_minus = np.cumsum(results_minus, axis=1)
                
                
                sens = (results_plus - results_minus)/2
                
                
                ave = np.average(sens, axis=0)
                perc5 = np.percentile(sens,5,axis=0)
                perc25 = np.percentile(sens,25,axis=0)
                perc50 = np.percentile(sens,50,axis=0)
                perc75 = np.percentile(sens,75,axis=0)
                perc95 = np.percentile(sens,95,axis=0)
                
                
                #plt.plot(ave, linewidth=2, color='darkblue')
                plt.plot(perc5, linewidth=1, linestyle=":", color='black')
                plt.plot(perc25, linewidth=1, color='black')
                plt.plot(perc50, linewidth=2, color='black')
                plt.plot(perc75, linewidth=1, color='black')
                plt.plot(perc95, linewidth=1, linestyle=":", color='black')
                plt.title(pca_allattr[i])
                plt.grid()
                plt.show()
                singular_sensitivity.append(np.average((results_plus - results_minus)/2))
            print(singular_sensitivity)
            print(keep_columns)                
            
            

        
        ## Sensitivity for RNN input channel
        
        X_test_m_plus = [X_test_RNN_m + 0.1, X_test_MLP]
        results_plus = scaler_y.inverse_transform(model.predict(X_test_m_plus))
        
        X_test_m_minus = [X_test_RNN_m - 0.1, X_test_MLP]
        results_minus = scaler_y.inverse_transform(model.predict(X_test_m_minus))
        
        # if target in convert_to_diff:
        #     results_plus = np.cumsum(results_plus, axis=1)
        #     results_minus = np.cumsum(results_minus, axis=1)

        
        sens = (results_plus - results_minus)/2
        
        
        ave = np.average(sens, axis=0)
        perc5 = np.percentile(sens,5,axis=0)
        perc25 = np.percentile(sens,25,axis=0)
        perc50 = np.percentile(sens,50,axis=0)
        perc75 = np.percentile(sens,75,axis=0)
        perc95 = np.percentile(sens,95,axis=0)
        
        
        #plt.plot(ave, linewidth=2, color='darkblue')
        plt.plot(perc5, linewidth=1, linestyle=":", color='black')
        plt.plot(perc25, linewidth=1, color='black')
        plt.plot(perc50, linewidth=2, color='black')
        plt.plot(perc75, linewidth=1, color='black')
        plt.plot(perc95, linewidth=1, linestyle=":", color='black')
        plt.title("RNN Input sensitivity, full channel")
        plt.grid()
        plt.show()
        
        singular_sens_input = []
        for i in range(len(X_test_RNN_m[0])):
            print(".", end="")
            X_test_RNN_m_plus = X_test_RNN_m.copy()
            X_test_RNN_m_plus[:,i] = X_test_RNN_m_plus[:,i] + 0.1
            X_test_m_plus = [X_test_RNN_m_plus, X_test_MLP]
            results_plus = scaler_y.inverse_transform(model.predict(X_test_m_plus))
            
            X_test_RNN_m_minus = X_test_RNN_m.copy()
            X_test_RNN_m_minus[:,i] = X_test_RNN_m_minus[:,i] - 0.1
            X_test_m_minus = [X_test_RNN_m_minus, X_test_MLP]
            results_minus = scaler_y.inverse_transform(model.predict(X_test_m_minus))
            
            sens = (results_plus - results_minus)/2
            
            singular_sens_input.append(np.percentile(sens,50, axis=0))
        
        
        vspread = np.max(np.abs(singular_sens_input))
        sns.heatmap(np.rot90(singular_sens_input), vmin = -vspread, vmax = vspread,
                    cmap="vlag")
        plt.show()
        
        plt.plot(np.mean(singular_sens_input, axis=0))
        plt.show()
        plt.plot(np.mean(singular_sens_input, axis=1))
        plt.show()
            
            

    # Plots
    if plot_samples==True:
        pred = scaler_y.inverse_transform(model.predict(X_train_m, verbose=0))
        
        if target in convert_to_diff:
            
            xtr = np.cumsum(scaler_y.inverse_transform(X_train_RNN), axis=1)
            off = np.rot90(np.tile(xtr[:,-1], (len(pred[0]),1) ), 3)
            
            pred = np.cumsum(pred, axis=1) + off
            ytr = np.cumsum(scaler_y.inverse_transform(y_train_RNN), axis=1) + off
        else:
            xtr = scaler_y.inverse_transform(X_train_RNN)
            ytr = scaler_y.inverse_transform(y_train_RNN)
        
        
        for i in range(10):
            s = np.random.randint(0, len(y_train_RNN))
            
            x = np.arange(0,len(X_train_RNN[0]),1)
            
            plt.title('Train')
            plt.plot(x, xtr[s], label='RNN input')
            
            
            x = np.arange(len(X_train_RNN[0]), len(X_train_RNN[0]) + len(y_train_RNN[0]),1)
            plt.plot(x,ytr[s], label='RNN output, true')
            
            
            
            plt.plot(x,pred[s], label='RNN output, predicted')
            plt.legend()
            
            
            plt.show()
        
        pred = scaler_y.inverse_transform(model.predict(X_test_m))
        
        if target in convert_to_diff:
            
            xts = np.cumsum(scaler_y.inverse_transform(X_test_RNN), axis=1)
            off = np.rot90(np.tile(xts[:,-1], (len(pred[0]),1) ), 3)
            
            pred = np.cumsum(pred, axis=1) + off
            yts = np.cumsum(scaler_y.inverse_transform(y_test_RNN), axis=1) + off
        else:
            xts = scaler_y.inverse_transform(X_test_RNN)
            yts = scaler_y.inverse_transform(y_test_RNN)
        
        
        for i in range(5):
            s = np.random.randint(0, len(y_test_RNN))
            
            x = np.arange(0,len(X_test_RNN[0]),1)
            
            plt.plot(x, xts[s], label='RNN input')
            
            x = np.arange(len(X_test_RNN[0]), len(X_test_RNN[0]) + len(y_test_RNN[0]),1)
            plt.plot(x,yts[s], label='RNN output, true')
            
            
            plt.title('test')
            plt.plot(x,pred[s], label='RNN output, predicted')
            plt.legend()
            plt.show()
    if target in convert_to_diff:
        truth = np.cumsum(y_test_RNN/target_lcs_correction/scaler_y.scale_, axis=1)
        pred = model.predict(X_test_m)
        pred = np.cumsum(pred/target_lcs_correction/scaler_y.scale_, axis=1)
        
    else:
        truth = y_test_RNN/target_lcs_correction/scaler_y.scale_
        pred = model.predict(X_test_m)
        pred = pred/target_lcs_correction/scaler_y.scale_

    if np.isnan(result_test):
        result_test = 0
    #print(-np.log10(result_test))
    
    if PCA_n != -1:
        keep_columns = pca_allattr
    
    print(f'MAE: {np.average(np.abs(truth-pred))}')
    return truth, pred, keep_columns, -np.log10(result_test)


#%%
# Tuning

# def optimize_me(hGRU,
#             hDrop1,
#             hDrop2, 
#             hDrop3,
#             hDense1,
#             hDense2,
#             hDense3,
#             hDense4,
#             hDense5):
#     truths = []
#     preds = []
#     quals = []
#     for i in np.arange(0.2,0.85,0.01):
#         print(f'Working on {i*100}%')
#         truth, pred, qual = run_me(i, hGRU,
#             hDrop1,
#             hDrop2, 
#             hDrop3,
#             hDense1,
#             hDense2,
#             hDense3,
#             hDense4,
#             hDense5,
#             asel_choice='pca')
#         #print(f'Quality is {qual:.2f}')
#         truths.append(truth)
#         preds.append(pred)
#         quals.append(qual)
        
        
#     diffs = []
    
#     for i in range(len(truths)):
#         diffs.append(np.average(np.abs(truths[i] - preds[i]), axis=0))
        

#     print(-np.average(diffs))
#     return truths, preds, quals, diffs



