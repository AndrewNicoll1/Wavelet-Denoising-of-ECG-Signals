import sys
import pywt
import numpy as np
import matplotlib.pyplot as plt
import peakutils
np.seterr(divide='ignore')
np.set_printoptions(threshold=sys.maxsize)
from SSIM_PIL import compare_ssim
from scipy.misc import electrocardiogram
from PIL import Image

#Number of repeats for random noise
repeats=1

#Amplitude of 50Hz Noise compared to peak height of ECG/MCG
amp=0.5

###Wavelet
##pywt.wavelist(None, kind='discrete')
##wavelet = 'db1'
###Decomposition level
##k=3

#Std range of random noise
amplitudes=[]
#numbers=[i for i in np.arange(0,1,0.02)]
numbers=[0.25]
for i in range(repeats):
    amplitudes.extend(numbers)

#Initialising lists
denoised_cak=[]
lambda_l_approx=[]
lambda_H_approx=[]
sparcity=[]
mean_cd=[]
std_cd=[]
kLmin_all_cd=[]
kHmin_all_cd=[]
sjL_all_cd=[]
sjH_all_cd=[]
sjL_all_ca=[]
sjH_all_ca=[]
cd_all=[]
ca_all=[]
sp=[]
RMSE=[]
SSIM=[]
all_snr=[]
all_snr2=[]

for a in amplitudes:

    #Clearing lists for each iteration in amplitudes   
    denoised_cak.clear()
    lambda_l_approx.clear()
    lambda_H_approx.clear()
    sparcity.clear()
    mean_cd.clear()
    std_cd.clear()
    kLmin_all_cd.clear()
    kHmin_all_cd.clear()
    sjL_all_cd.clear()
    sjH_all_cd.clear()
    sjL_all_ca.clear()
    sjH_all_ca.clear()
    cd_all.clear()
    ca_all.clear()
    sp.clear()

    #50Hz Signal
    Fs=360
    f=50
    sample=600
    x=np.arange(sample)
    noise=np.sin(2*np.pi*f*x/Fs)

    #ECG signal
    ecg=electrocardiogram().tolist()
    ecg_short=ecg[5000:5600]
    signal=np.array(ecg_short)
    time = np.arange(signal.size)/Fs
    
    #Std of random noise
    sigma=a
    rand_noise=np.random.normal(0, a, len(signal))

    #Baseline value and normalising 50Hz amplitude
    baseline=peakutils.baseline(signal)
    signal_peak=max(signal)
    signal=signal.tolist()
    
    peak_pos=signal.index(signal_peak)
    point=baseline[peak_pos]
    difference=np.abs(signal_peak-point)

    #Noisy signal (50Hz+Random)
    fifty_noise=(difference*amp*noise)
    random_noise=rand_noise
    sig_noisy=signal+fifty_noise+random_noise

    #Signal-to-noise ratio for 50Hz
    rms_noise=np.sqrt(np.square(fifty_noise).mean())
    SNR=signal_peak/rms_noise
    all_snr.append(SNR)

    #Signal-to-noise ratio for white noise
    rms_noise2=np.sqrt(np.square(random_noise).mean())
    SNR2=signal_peak/rms_noise2
    all_snr2.append(SNR2)

    #Maximum decomposition level
    l=len(sig_noisy)
    max_level=pywt.swt_max_level(l)
    print("Max Level: {}".format(max_level))

    denoised_cd=[[] for i in range(max_level)]

    #Decomposition level selection
    if SNR<1.4:
        wavelet='db1'
        k=3
    else:
        if SNR>4.6:
            wavelet='db38'
            if SNR2>6.9:
                k=2
            else:
                k=3
        else:
            if SNR>2.8:
                wavelet='db38'
                if SNR2>8.6:
                    k=2
                else:
                    k=3
            else:
                if SNR>1.9:
                    wavelet='db38'
                    if SNR2>11.6:
                        k=2
                    else:
                        k=3
                else:
                    wavelet='db38'
                    k=3

    #Stationary Wavelet Transform
    coeffs=pywt.swt(sig_noisy, wavelet, level=max_level, trim_approx=False, norm=True) 

    for i in range(max_level):

        cd_all.append(coeffs[i][1])
        ca_all.append(coeffs[i][0])
        
        #Peak-to-sum ratio for each decomposotion level
        sj=np.max(np.abs(coeffs[i][1]))/np.sum(np.abs(coeffs[i][1]))  
        sp.append(sj)
        
        #Mean and s.t.d of jth detail component
        for l in range(len(coeffs[i][1])):
            mean=np.mean(coeffs[i][1])
            std=np.std(coeffs[i][1])

        mean_cd.append(mean)
        std_cd.append(std)

        #Min values of kj,l and kj,h for detail
        kLmin=(mean-np.abs(np.min(coeffs[i][1])))/std
        kHmin=(np.abs(np.max(coeffs[i][1]))-mean)/std

        kLmin_all_cd.append(kLmin)
        kHmin_all_cd.append(kHmin)

        #Peak-to_sum ratio of positive and negative detail coefficients for each decomosition level
        sjL=np.abs(np.min(coeffs[i][1]))/np.abs(sum((l for l in coeffs[i][1] if l<0)))
        sjH=np.abs(np.max(coeffs[i][1]))/np.abs(sum((l for l in coeffs[i][1] if l>=0)))

        sjL_all_cd.append(sjL)
        sjH_all_cd.append(sjH)
        
        #Peak-to_sum ratio of positive and negative approximation coefficients for each decomosition level
        asjL=np.abs(np.min(coeffs[i][0]))/np.abs(sum((l for l in coeffs[i][0] if l<0)))
        asjH=np.abs(np.max(coeffs[i][0]))/np.abs(sum((l for l in coeffs[i][0] if l>=0)))

        sjL_all_ca.append(asjL)
        sjH_all_ca.append(asjH)

    #Data up to kth level
    sparcity=sp[-k:]
    cd_all_1_k2=cd_all[-k:]
    ca_k=ca_all[(max_level-k)]
    mean_cd_1_k=mean_cd[-k:]
    std_cd_1_k=std_cd[-k:]
    kLmin_all_cd_1_k=kLmin_all_cd[-k:]
    kHmin_all_cd_1_k=kHmin_all_cd[-k:]
    sjL_all_cd_1_k=sjL_all_cd[-(k+1):]
    sjH_all_cd_1_k=sjH_all_cd[-(k+1):]
    sjL_all_ca_1_k=sjL_all_ca[-(k+1):]
    sjH_all_ca_1_k=sjH_all_ca[-(k+1):]

    #Reference Peak-to-sum ratio detailed for k and k+1
    srL_cd=0.5*(sjL_all_cd_1_k[1]+sjL_all_cd_1_k[0])
    srH_cd=0.5*(sjH_all_cd_1_k[1]+sjH_all_cd_1_k[0])

    #Reference Peak-to-sum ratio Approximate for k and k+1
    srL_ca=0.5*(sjL_all_ca_1_k[1]+sjL_all_ca_1_k[0])
    srH_ca=0.5*(sjH_all_ca_1_k[1]+sjH_all_ca_1_k[0])
    cd_all_1_k=cd_all_1_k2[:]

    #Denoising Detail Coefficients
    for idx, val in enumerate(sparcity):
        if 0<val<=0.01:
            lambda_L1=(mean_cd_1_k[idx]-kLmin_all_cd_1_k[idx]*std_cd_1_k[idx])
            lambda_H1=(mean_cd_1_k[idx]+kHmin_all_cd_1_k[idx]*std_cd_1_k[idx])
            for i in cd_all_1_k[idx]:
                if i >=0:
                    coeff1 = pywt.threshold(cd_all_1_k[idx], lambda_H1,'hard')
                    cd_all_1_k[idx] = coeff1
                else:
                    coeff1 = pywt.threshold(cd_all_1_k[idx], lambda_L1,'hard')
                    cd_all_1_k[idx] = coeff1
        elif 0.01<val<0.2:
            lambda_L2=(mean_cd_1_k[idx]-(1-(sjL_all_cd_1_k[idx]/srL_cd))*kLmin_all_cd_1_k[idx]*std_cd_1_k[idx])    
            lambda_H2=(mean_cd_1_k[idx]+(1-(sjH_all_cd_1_k[idx]/srH_cd))*kHmin_all_cd_1_k[idx]*std_cd_1_k[idx])    
            for i in cd_all_1_k[idx]:
                if i >=0:
                    coeff2 = pywt.threshold(cd_all_1_k[idx], lambda_H2,'hard')
                    cd_all_1_k[idx] = coeff2
                else:
                    coeff2 = pywt.threshold(cd_all_1_k[idx], lambda_L2,'hard')
                    cd_all_1_k[idx] = coeff2


    #Denoising kth approximation component
    Approx_k_denoised=ca_k[:]
    ap_mean=np.mean(ca_k)
    ap_std=np.std(ca_k)
    ap_kL_min=(ap_mean-np.abs(np.max(ca_k[:])))/ap_std
    ap_kH_min=(np.abs(np.min(ca_k[:]))-ap_mean)/ap_std
    kj_L_ap = (1-(sjL_all_ca_1_k[1])/(srL_ca))*ap_kL_min
    kj_H_ap = (1-(sjH_all_ca_1_k[1])/(srH_ca))*ap_kH_min
    lambda_kL_approx = ap_mean-kj_L_ap*ap_std
    lambda_kH_approx = ap_mean+kj_H_ap*ap_std
    for i in range(len(Approx_k_denoised)):
        if i>=0:
            coeffk_approx = pywt.threshold(Approx_k_denoised[i],lambda_kH_approx,'hard')
            Approx_k_denoised[i] = coeffk_approx
        else:
            coeffk_approx = pywt.threshold(Approx_k_denoised[i],lambda_kL_approx,'hard')
            Approx_k_denoised[i] = coeffk_approx

    #Formatting coefficient lists for the iswt
    Approx_k_denoised=Approx_k_denoised.tolist()
    Approx_k_denoised=[np.array(Approx_k_denoised)]
    a=len(cd_all_1_k[0])
    new_ca=[np.array([None]*a) for i in range(k-1)]
    Approx_k_denoised.extend(new_ca)

    new_ca=Approx_k_denoised
    new_cd=cd_all_1_k

    def merge(list1, list2): 
        merged_list = [(list1[i], list2[i]) for i in range(0, len(list2))] 
        return merged_list

    final=merge(new_ca,new_cd)

    #Inverse stationary Wavelet Transform
    denoised_signal=pywt.iswt(final,wavelet,norm=True)

    #RMSE of signal and denoised signal
    difference_array = np.subtract(signal, denoised_signal)
    squared_array = np.square(difference_array)
    root_mse = np.sqrt((squared_array.mean())/len(signal))
    RMSE.append(root_mse)

    plt.ylim([-1.5,2.6])
    plt.axis("off")
    plt.plot(signal, 'r')
    plt.savefig("signal_2.png")
    plt.clf()
    plt.ylim([-1.5,2.6])
    plt.axis("off")
    plt.plot(denoised_signal, 'r')
    plt.savefig("denoised_2.png")

    #SSIM of signal and denoised signal
    image1 = Image.open("signal_2.png")
    image2 = Image.open("denoised_2.png")
    value = compare_ssim(image1, image2)
    print("50Hz SNR: {} White Noise SNR: {} Sigma: {} RMSE: {}".format(SNR, SNR2, sigma, root_mse))
    SSIM.append(value)

##    file=open("UNDEC RMSE 50HZ.txt", "a")
##    file.write(str(root_mse))
##    file.write("\n")
##    file.close()
##
##    file=open("UNDEC SSIM 50HZ.txt", "a")
##    file.write(str(value))
##    file.write("\n")
##    file.close()

    #Plot 50Hz noise
    plt.clf()
    plt.axis("on")
    plt.plot(time, sig_noisy, 'b', label="Noisy Signal")
    plt.plot(time, signal, 'y', label="Original Signal", linewidth=2)
    plt.plot(time, denoised_signal, 'r', label="Denoised Signal", linewidth=2)
    #plt.plot(time, baseline, 'k')
    plt.legend(prop={'size':14})
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Voltage (mV)", fontsize=14)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', direction='out')
    plt.tick_params(axis='y', which='minor', direction='out')
    plt.show()

    #Clearing lists
    denoised_cd.clear()
    plt.clf()
    plt.close()

print("\n")
rmse_mean=np.mean(RMSE)
ssim_mean=np.mean(SSIM)
rmse_std=np.std(RMSE)
ssim_std=np.std(SSIM)
snr_mean=np.mean(all_snr)
snr2_mean=np.mean(all_snr2)


print("Mean RMSE: {} STD of RMSE: {}".format(rmse_mean, rmse_std))
print("Mean SSIM: {} STD of SSIM: {}".format(ssim_mean, ssim_std))
print("Mean 50Hz SNR: {}".format(snr_mean))
print("Mean White Noise SNR: {}".format(snr2_mean))


n=len(RMSE)//repeats
splt_rmse=[RMSE[i:i+n] for i in range(0, len(RMSE), n)]
splt_ssim=[SSIM[i:i+n] for i in range(0, len(SSIM), n)]

mean_rmse=np.mean(splt_rmse, axis=0)
std_rmse=np.std(splt_rmse, axis=0)
mean_ssim=np.mean(splt_ssim, axis=0)
std_ssim=np.std(splt_ssim, axis=0)

rmse_error=[i/np.sqrt(repeats) for i in std_rmse]
ssim_error=[i/np.sqrt(repeats) for i in std_ssim]

 
color='red'
fig, ax1 = plt.subplots()
ax1.set_xlabel('Standard Deviation')
ax1.set_ylabel('RMSE', color=color)
ax1.plot(numbers, mean_rmse, 'ro-', markersize=4)
ax1.tick_params(axis='y', labelcolor=color)
plt.errorbar(numbers, mean_rmse, yerr=rmse_error, ecolor='black', elinewidth=1, capsize=1)
color='blue'
ax2 = ax1.twinx()  
ax2.set_ylabel('SSIM', color=color)  
ax2.plot(numbers, mean_ssim, 'bo-', markersize=4)
ax2.tick_params(axis='y', labelcolor=color)
plt.errorbar(numbers, mean_ssim, yerr=ssim_error, ecolor='black', elinewidth=1, capsize=1)
plt.title('Undecimated \n 50Hz of amplitude {} & random noise'.format(amp))
#plt.show()

#Comparing UNDEC to DEC

dec_rmse=[]
dec_ssim=[]

f=open('RMSE.txt',"r")
for line in f:
    dec_rmse.append(float(line.strip('\n')))

f=open('SSIM.txt',"r")
for line in f:
    dec_ssim.append(float(line.strip('\n')))

n=len(RMSE)//repeats
splt_DEC_rmse=[dec_rmse[i:i+n] for i in range(0, len(dec_rmse), n)]
splt_DEC_ssim=[dec_ssim[i:i+n] for i in range(0, len(dec_ssim), n)]

DEC_mean_rmse=np.mean(splt_DEC_rmse, axis=0)
DEC_std_rmse=np.std(splt_DEC_rmse, axis=0)
DEC_mean_ssim=np.mean(splt_DEC_ssim, axis=0)
DEC_std_ssim=np.std(splt_DEC_ssim, axis=0)

DEC_rmse_error=[i/np.sqrt(repeats) for i in DEC_std_rmse]
DEC_ssim_error=[i/np.sqrt(repeats) for i in DEC_std_ssim]

plt.figure()
plt.xlabel("Standard Deviation", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.plot(numbers, DEC_mean_rmse, 'bo-', markersize=4, label="Decimated")
plt.plot(numbers, mean_rmse, 'ro-', markersize=4, label="Undecimated")
plt.errorbar(numbers, mean_rmse, yerr=rmse_error, ecolor='black', elinewidth=1, capsize=3)
plt.errorbar(numbers, DEC_mean_rmse, yerr=DEC_rmse_error, ecolor='black', elinewidth=1, capsize=3)
#plt.title('Undecimated vs Decimated \n ({} for {} Repeats)'.format(wavelet, repeats))
plt.legend(prop={'size':14})
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', direction='out')
plt.tick_params(axis='y', which='minor', direction='out')
plt.show()

plt.figure()
plt.xlabel("Standard Deviation")
plt.ylabel("SSIM")
plt.plot(numbers, mean_ssim, 'ro-', markersize=4, label="Undecimated")
plt.errorbar(numbers, mean_ssim, yerr=ssim_error, ecolor='black', elinewidth=1, capsize=3)
plt.plot(numbers, DEC_mean_ssim, 'bo-', markersize=4, label="Decimated")
plt.errorbar(numbers, DEC_mean_ssim, yerr=DEC_ssim_error, ecolor='black', elinewidth=1, capsize=3)
plt.title('Undecimated vs Decimated \n ({} for {} Repeats)'.format(wavelet, repeats))
plt.legend()
#plt.show()

##plt.clf()
##fig, axarr = plt.subplots(nrows=3, ncols=2)
##for ii in range(3):
##    axarr[ii, 0].plot(cd_all_1_k2[ii], 'b')
##    axarr[ii, 1].plot(new_cd[ii], 'r')
##    #axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14)
##    if ii == 0:
##        axarr[ii, 0].set_title("Detail Coefficients", fontsize=24)
##        axarr[ii, 1].set_title("Denoised Detail Coefficients", fontsize=24)
##
##plt.show()

