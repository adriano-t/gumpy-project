#mlxtend
#conda install pywavelets

import sys, os, os.path
sys.path.append('./gumpy')
import numpy as np
import gumpy
grazb_base_dir='./grazdata'
subject = 'B04'
grazb_data=gumpy.data.GrazB(grazb_base_dir, subject)
#load the dataset
grazb_data.load()
grazb_data.print_stats()


if 0:
    grazb_data.raw_data[:, 0] -= 2 * grazb_data.raw_data[:, 1]
    grazb_data.raw_data[:, 2] -= 2 * grazb_data.raw_data[:, 2]


# this returns a butter-bandpass filtered version of the entire dataset
btr_data = gumpy.signal.butter_bandpass(grazb_data, lo=2, hi=60)

# it is also possible to use filters on individual electrodes using 
# the .raw_data field of a dataset. The example here will remove a certain
# from a single electrode using a Notch filter. This example also demonstrates
# that parameters will be forwarded to the internal call to the filter, in this
# case the scipy implementation iirnotch (Note that iirnotch is only available
# in recent versions of scipy, and thus disabled in this example by default)

# frequency to be removed from the signal
if False:
    f0 = 50.0 
    # quality factor
    Q = 50.0  
    # get the cutoff frequency
    w0 = f0/(grazb_data.sampling_freq/2) 
    # apply the notch filter
    notch_data = gumpy.signal.notch(grazb_data.raw_data[:, 0], w0, Q)


''' Normalization '''
# normalize the data first
norm_data = gumpy.signal.normalize(grazb_data, 'mean_std')
# let's see some statistics
print("""Normalized Data:
  Mean    = {:.3f}
  Min     = {:.3f}
  Max     = {:.3f}
  Std.Dev = {:.3f}""".format(
  np.nanmean(norm_data),np.nanmin(norm_data),np.nanmax(norm_data),np.nanstd(norm_data)
))


''' Plotting and Feature Extraction '''

#%matplotlib inline
import matplotlib.pyplot as plt 

# Plot after filtering with a butter bandpass (ignore normalization)
plt.figure()
plt.clf()
plt.plot(btr_data[grazb_data.trials[0]: grazb_data.trials[1], 0], label='C3')
plt.plot(btr_data[grazb_data.trials[0]: grazb_data.trials[1], 1], alpha=0.7, label='C4')
plt.plot(btr_data[grazb_data.trials[0]: grazb_data.trials[1], 2], alpha=0.7, label='Cz')
plt.legend()
plt.title(" Filtered Data")


'''EEG band visualization'''

# determine the trial that we wish to plot
n_trial = 120
# now specify the alpha and beta cutoff frequencies
lo_a, lo_b = 7, 16
hi_a, hi_b = 13, 24

# first step is to filter the data
flt_a = gumpy.signal.butter_bandpass(grazb_data, lo=lo_a, hi=hi_a)
flt_b = gumpy.signal.butter_bandpass(grazb_data, lo=lo_b, hi=hi_b)

# finally we can visualize the data
gumpy.plot.EEG_bandwave_visualizer(grazb_data, flt_a, n_trial, lo_a, hi_a)
gumpy.plot.EEG_bandwave_visualizer(grazb_data, flt_b, n_trial, lo_a, hi_a)



'''Extract Trials'''
# retrieve the trials from the filtered data. This requires that the function
# knows the number of trials, labels, etc. when only passed a (filtered) data matrix
trials = grazb_data.trials
labels = grazb_data.labels
sampling_freq = grazb_data.sampling_freq
data_class_a = gumpy.utils.extract_trials(flt_a, trials=trials, labels=labels, sampling_freq=sampling_freq)

# it is also possible to pass an instance of Dataset and filtered data.
# gumpy will then infer all necessary details from the dataset
data_class_b = gumpy.utils.extract_trials(grazb_data, flt_b)

# similar to other functions, this one allows to pass an entire instance of Dataset
# to operate on the raw data
data_class1 = gumpy.utils.extract_trials(grazb_data)



'''Visualize the classes'''
# specify some cutoff values for the visualization
lowcut_a, highcut_a = 14, 30
# and also an interval to display
interval_a = [0, 8]
# visualize logarithmic power?
logarithmic_power = False

# visualize the extracted trial from above
gumpy.plot.average_power(data_class_a, lowcut_a, highcut_a, interval_a, grazb_data.sampling_freq, logarithmic_power)


'''Wavelet Transform'''
# As with most functions, you can pass arguments to a 
# gumpy function that will be forwarded to the backend.
# In this example the decomposition levels are mandatory, and the 
# mother wavelet that should be passed is optional
level = 6
wavelet = 'db4'

# now we can retrieve the dwt for the different channels
mean_coeff_ch0_c1 = gumpy.signal.dwt(data_class1[0], level=level, wavelet=wavelet)
mean_coeff_ch1_c1 = gumpy.signal.dwt(data_class1[1], level=level, wavelet=wavelet)
mean_coeff_ch0_c2 = gumpy.signal.dwt(data_class1[3], level=level, wavelet=wavelet)
mean_coeff_ch1_c2 = gumpy.signal.dwt(data_class1[4], level=level, wavelet=wavelet)

# gumpy's signal.dwt function returns the approximation of the 
# coefficients as first result, and all the coefficient details as list
# as second return value (this is contrast to the backend, which returns
# the entire set of coefficients as a single list)
approximation_C3 = mean_coeff_ch0_c2[0]
approximation_C4 = mean_coeff_ch1_c2[0]

# as mentioned in the comment above, the list of details are in the second
# return value of gumpy.signal.dwt. Here we save them to additional variables
# to improve clarity
details_c3_c1 = mean_coeff_ch0_c1[1]
details_c4_c1 = mean_coeff_ch1_c1[1]
details_c3_c2 = mean_coeff_ch0_c2[1]
details_c4_c2 = mean_coeff_ch1_c2[1]

# gumpy exhibits a function to plot the dwt results. You must pass three lists,
# i.e. the labels of the data, the approximations, as well as the detailed coeffs,
# so that gumpy can automatically generate appropriate titles and labels.
# you can pass an additional class string that will be incorporated into the title.
# the function returns a matplotlib axis object in case you want to further
# customize the plot.
gumpy.plot.dwt(
    [approximation_C3, approximation_C4],
    [details_c3_c1, details_c4_c1],
    ['C3, c1', 'C4, c1'],
    level, grazb_data.sampling_freq, 'Class: Left')


'''DWT reconstruction and visualization'''

gumpy.plot.reconstruct_without_approx(
    [details_c3_c2[4], details_c4_c2[4]], 
    ['C3-c1', 'C4-c1'], level=6)

gumpy.plot.reconstruct_without_approx(
    [details_c3_c1[5], details_c4_c1[5]], 
    ['C3-c1', 'C4-c1'], level=6)

gumpy.plot.reconstruct_with_approx(
    [details_c3_c1[5], details_c4_c1[5]],
    ['C3', 'C4'], wavelet=wavelet)


'''Welch's Power Spectral Density estimate'''
# the function gumpy.plot.welch_psd returns the power densities as 
# well as a handle to the figure. You can also pass a figure in if you 
# wish to modify the plot
fig = plt.figure()
plt.title('Customized plot')
ps, fig = gumpy.plot.welch_psd(
    [details_c3_c1[4], details_c3_c2[4]],
    ['C3 - c1', 'C4 - c1'],
    grazb_data.sampling_freq, fig=fig)

ps, fig = gumpy.plot.welch_psd(
    [details_c4_c1[4], details_c4_c2[4]],
    ['C3 - c1', 'C4 - c1'],
    grazb_data.sampling_freq)



'''Alpha and Beta sub-bands'''
def alpha_subBP_features(data):
    # filter data in sub-bands by specification of low- and high-cut frequencies
    alpha1 = gumpy.signal.butter_bandpass(data, 8.5, 11.5, order=6)
    alpha2 = gumpy.signal.butter_bandpass(data, 9.0, 12.5, order=6)
    alpha3 = gumpy.signal.butter_bandpass(data, 9.5, 11.5, order=6)
    alpha4 = gumpy.signal.butter_bandpass(data, 8.0, 10.5, order=6)

    # return a list of sub-bands
    return [alpha1, alpha2, alpha3, alpha4]

alpha_bands = np.array(alpha_subBP_features(grazb_data))


def beta_subBP_features(data):
    beta1 = gumpy.signal.butter_bandpass(data, 14.0, 30.0, order=6)
    beta2 = gumpy.signal.butter_bandpass(data, 16.0, 17.0, order=6)
    beta3 = gumpy.signal.butter_bandpass(data, 17.0, 18.0, order=6)
    beta4 = gumpy.signal.butter_bandpass(data, 18.0, 19.0, order=6)
    return [beta1, beta2, beta3, beta4]

beta_bands = np.array(beta_subBP_features(grazb_data))



'''Feature extraction using sub-bands'''

# Method 1: logarithmic sub-band power
def powermean(data, trial, fs, w):
    return np.power(data[trial+fs*4+w[0]: trial+fs*4+w[1],0],2).mean(), \
           np.power(data[trial+fs*4+w[0]: trial+fs*4+w[1],1],2).mean(), \
           np.power(data[trial+fs*4+w[0]: trial+fs*4+w[1],2],2).mean()

def log_subBP_feature_extraction(alpha, beta, trials, fs, w):
    # number of features combined for all trials
    n_features = 15
    # initialize the feature matrix
    X = np.zeros((len(trials), n_features))
    
    # Extract features
    for t, trial in enumerate(trials):
        power_c31, power_c41, power_cz1 = powermean(alpha[0], trial, fs, w)
        power_c32, power_c42, power_cz2 = powermean(alpha[1], trial, fs, w)
        power_c33, power_c43, power_cz3 = powermean(alpha[2], trial, fs, w)
        power_c34, power_c44, power_cz4 = powermean(alpha[3], trial, fs, w)
        power_c31_b, power_c41_b, power_cz1_b = powermean(beta[0], trial, fs, w)
        
        X[t, :] = np.array(
            [np.log(power_c31), np.log(power_c41), np.log(power_cz1),
             np.log(power_c32), np.log(power_c42), np.log(power_cz2),
             np.log(power_c33), np.log(power_c43), np.log(power_cz3), 
             np.log(power_c34), np.log(power_c44), np.log(power_cz4),
             np.log(power_c31_b), np.log(power_c41_b), np.log(power_cz1_b)])

    return X

w1 = [0,125]
w2 = [125,250]

features1 = log_subBP_feature_extraction(
    alpha_bands, beta_bands, 
    grazb_data.trials, grazb_data.sampling_freq,
    w1)

features2 = log_subBP_feature_extraction(
    alpha_bands, beta_bands, 
    grazb_data.trials, grazb_data.sampling_freq,
    w2)                                          

# concatenate the features and normalize the data
features = np.concatenate((features1.T, features2.T)).T
features -= np.mean(features)
features = gumpy.signal.normalize(features, 'min_max')

# print shape to quickly check if everything is as expected
features.shape

# Method 2: DWT
def dwt_features(data, trials, level, sampling_freq, w, n, wavelet): 
    import pywt
    
    # number of features per trial
    n_features = 9 
    # allocate memory to store the features
    X = np.zeros((len(trials), n_features))

    # Extract Features
    for t, trial in enumerate(trials):
        signals = data[trial + fs*4 + (w[0]) : trial + fs*4 + (w[1])]
        coeffs_c3 = pywt.wavedec(data = signals[:,0], wavelet=wavelet, level=level)
        coeffs_c4 = pywt.wavedec(data = signals[:,1], wavelet=wavelet, level=level)
        coeffs_cz = pywt.wavedec(data = signals[:,2], wavelet=wavelet, level=level)

        X[t, :] = np.array([
            np.std(coeffs_c3[n]), np.mean(coeffs_c3[n]**2),  
            np.std(coeffs_c4[n]), np.mean(coeffs_c4[n]**2),
            np.std(coeffs_cz[n]), np.mean(coeffs_cz[n]**2), 
            np.mean(coeffs_c3[n]),
            np.mean(coeffs_c4[n]), 
            np.mean(coeffs_cz[n])])
        
    return X
# We'll work with the data that was postprocessed using a butter bandpass
# filter further above

# to see it work, enable here. We'll use the log-power features further
# below, though
if False:
    w = [0, 256]
    
    # extract the features
    trials = grazb_data.trials
    fs = grazb_data.sampling_freq
    features1= np.array(dwt_features(btr_data, trials, 5, fs, w, 3, "db4"))
    features2= np.array(dwt_features(btr_data, trials, 5, fs, w, 4, "db4"))

    # concat the features and normalize
    features = np.concatenate((features1.T, features2.T)).T
    features -= np.mean(features)
    features = gumpy.signal.normalize(features, 'min_max')



'''Sequential Feature Selection Algorithm'''
feature_idx, cv_scores, algorithm = gumpy.features.sequential_feature_selector(features, labels, 'LDA', (6, 12), 10, 'SFFS')
# let's see some information about the results
print('Selection Algorithm: ', algorithm)
print('Average Score:       ', cv_scores * 100)
print('Feature Index:       ', feature_idx)

'''PCA'''
PCA = gumpy.features.PCA_dim_red(features, 0.95)

'''Splitting data for training and classification'''

test_size = 0.2

# gumpy exposes several methods to split a dataset, as shown in the examples:
if 1: 
    split_features = np.array(gumpy.data.split(features, labels,test_size))
if 0: 
    n_splits=5
    split_features = np.array(gumpy.data.variable_cross_validation(features, labels, n_splits)) 
if 0: 
    split_features = np.array(gumpy.data.split(PCA, labels, test_size))

# the functions return a list with the data according to the following example
X_train = split_features[0]
X_test = split_features[1]
Y_train = split_features[2]
Y_test = split_features[3]

X_train.shape

'''Plotting features'''
gumpy.plot.PCA("3D", features, split_features[0], split_features[2])



'''Classification with normal split'''
results, clf = gumpy.classify('LDA', X_train, Y_train, X_test, Y_test)
print("Classification results on test set:")
print(results)
print("Accuracy: ", results.accuracy)


'''Confusion Matrix'''
gumpy.plot.confusion_matrix(Y_test, results.pred)


'''Voting Classifier with feature selection'''
result, _ = gumpy.classification.vote(X_train, Y_train, X_test, Y_test, 'hard', False, (6,12))
print("Classification result for hard voting classifier")
print(results)
print("Accuracy: ", results.accuracy)

gumpy.plot.confusion_matrix(result.pred, Y_test)

result, _ = gumpy.classification.vote(X_train, Y_train, X_test, Y_test, 'soft', True, (6,12))
print("Classification result for soft voting classifier")
print(results)
print("Accuracy: ", results.accuracy)
gumpy.plot.confusion_matrix(result.pred, Y_test)