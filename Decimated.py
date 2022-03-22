import numpy as np                  
import matplotlib.pyplot as plt     
import pywt                         
from pywt import wavedec
import pywt.data
np.seterr(divide='ignore')
from SSIM_PIL import compare_ssim
from PIL import Image
from scipy.misc import electrocardiogram
import peakutils

#db6 std>0.2, k=3->4, amp=0.2
#db6 std>0.15, k=3->4, amp=1
#db6 all std>, k=4, amp=2

repeats=10

#Amplitude of 50Hz noise
amp=1.2

q=1.3

amplitudes=[]

k=4

# Select a wavelet and print information about it.
wavelet = 'db6'

#Std of random noise
numbers=[i for i in np.arange(0,1,0.02)]
#numbers=[0.1]
for i in range(repeats):
    amplitudes.extend(numbers)

limit=0.4

Sj = [] 
Cd_all = []
Ca_all = []
mean_j = []
std_j = []
Kj_Lmin = []
Kj_Hmin = []
Sj_L = []
Sj_H = []
Sj_L_approx = []
Sj_H_approx = []
coeff_total = []
RMSE=[]
SSIM=[]

for a in amplitudes:

    #Decomposition level selection based on RMSE

##    if a<limit:
##        k=3
##    else:
##        k=4

    #50Hz Signal
    Fs=360
    f=50
    sample=600
    x=np.arange(sample)
    noise=np.sin(2*np.pi*f*x/Fs)

    sigma=a

    #ECG signal
    ecg=electrocardiogram().tolist()
    ecg_short=ecg[5000:5600]
    x_ecg=np.array(ecg_short)
    time = np.arange(x_ecg.size)/Fs

    #Baseline value and normalising 50Hz amplitude
    baseline=peakutils.baseline(x_ecg)
    signal_peak=max(x_ecg)
    signal=x_ecg.tolist()
    
    peak_pos=signal.index(signal_peak)
    point=baseline[peak_pos]
    difference=np.abs(signal_peak-point)
    
    # Generate random noise.
    mu=0 # mean and standard deviation
    Gaussian_noise = np.random.normal(mu, sigma, len(x_ecg))

    noise=(difference*amp*noise)+Gaussian_noise
    data = x_ecg+noise
    data1 = data[:]

    #Signal-to-noise ratio
    rms_noise=np.sqrt(np.square(noise).mean())
    SNR=signal_peak/rms_noise

    # Determine and print the maximum useful decomposition for a given data set.
    max_level = pywt.dwt_max_level(len(data), wavelet)
    print("Max Level: {}".format(max_level))

    Sj.clear() 
    Cd_all.clear()
    Ca_all.clear()
    mean_j.clear()
    std_j.clear()
    Kj_Lmin.clear()
    Kj_Hmin.clear()
    Sj_L.clear()
    Sj_H.clear()
    Sj_L_approx.clear()
    Sj_H_approx.clear()
    coeff_total.clear()

    for ii in range(max_level):
        (data, coeff_d) = pywt.dwt(data, wavelet)
        
        # Append all Detail and Approximation coefficient arrays.
        Cd_all.append(coeff_d)
        Ca_all.append(data)
        
        # Calculate Sparcity for given Decomposition level
        Sjj = np.max(np.abs(coeff_d))/np.sum(np.abs(coeff_d))
        Sj.append(Sjj)
            
        # Calculate the mean and s.t.d. for each detail level.
        for i in range(len(coeff_d)):
            mean_val = np.mean(coeff_d)
            std_val = np.std(coeff_d)
        # Append the mean and s.t.d.
        mean_j.append(mean_val)
        std_j.append(std_val)
        
        # Calculate the minimum values of Kj,L and Kj,H that cover all coefficients in the jth Detail component
        K_Lmin = (mean_val - np.abs(np.min(coeff_d)))/std_val
        K_Hmin = (np.abs(np.max(coeff_d))-mean_val)/std_val
        # Append K_Lmin and K_Hmin
        Kj_Lmin.append(K_Lmin)
        Kj_Hmin.append(K_Hmin)
        
        # Calculate the peak-to-sum ratios of the positive and negative coefficient values for jth detail level.
        val1 = np.abs(np.min(coeff_d))/np.abs(sum((p for p in coeff_d if p < 0)))
        val2 = np.abs(np.max(coeff_d))/np.abs(sum((p for p in coeff_d if p >= 0)))
        # Append values for use elsewhere
        Sj_L.append(val1)
        Sj_H.append(val2)
        
        # Calculate the peak-to-sum ratios of the positive and negative coefficient values for jth Approx level.
        val1_appox = np.abs(np.min(data))/np.abs(sum((p for p in data if p < 0)))
        val2_appox = np.abs(np.max(data))/np.abs(sum((p for p in data if p >= 0)))
        # Append values for use elsewhere
        Sj_L_approx.append(val1_appox)
        Sj_H_approx.append(val2_appox)
        

    ## Based on the sparcity we select k = 5

    coeff_d_reverse = Cd_all[::-1]
    coeff_a_reverse = Ca_all[::-1]
    S_j = Sj[::-1]
    Sj_L_rev = Sj_L[::-1]
    Sj_H_rev = Sj_H[::-1]
    Sj_L_approx_rev = Sj_L_approx[::-1]
    Sj_H_approx_rev = Sj_H_approx[::-1]
    mean_j_rev = mean_j[::-1]
    std_j_rev = std_j[::-1]
    Kj_Lmin_rev = Kj_Lmin[::-1]
    Kj_Hmin_rev = Kj_Hmin[::-1]

    # There are the arrays used for decomposition based on the level selected.
    Approx_k = coeff_a_reverse[-k]
    Detail_k_to_1 = coeff_d_reverse[-k:]
    sparcity = S_j[-k:]
    Sj_L_k_to_1 = Sj_L_rev[-(k):]
    Sj_H_k_to_1 = Sj_H_rev[-(k):]
    Sj_L_approx_k_to_1 = Sj_L_approx_rev[-(k):]
    Sj_H_approx_k_to_1 = Sj_H_approx_rev[-(k):]
    mean_k_to_1 = mean_j_rev[-k:]
    std_k_to_1 = std_j_rev[-k:]
    K_Lmin_k_to_1 = Kj_Lmin_rev[-k:]
    K_Hmin_k_to_1 = Kj_Hmin_rev[-k:]

    # Calculate reference peak-to-sum coefficient values for Detail components.
    # This is useful for 0.01 < Sj < Tr.
    Sr_L = 0.5*(Sj_L_rev[-(k+1)]+Sj_L_rev[-(k)])
    Sr_H = 0.5*(Sj_H_rev[-(k+1)]+Sj_H_rev[-(k)])

    # Calculate reference peak-to-sum coefficient values.
    # This is for the Approximation coefficient.
    Sr_L_cA = 0.5*(Sj_L_approx_rev[-(k+1)]+Sj_L_approx_rev[-(k)])
    Sr_H_cA = 0.5*(Sj_H_approx_rev[-(k+1)]+Sj_H_approx_rev[-(k)])

    coeff_d_tot = Detail_k_to_1[:]

    # Thresholding function for the detail coefficients.
    for idx, val in enumerate(sparcity):
        if 0<val<=0.01:
            Lambda_jL1 = (mean_k_to_1[idx]-1.5*K_Lmin_k_to_1[idx]*std_k_to_1[idx])
            Lambda_jH1 = (mean_k_to_1[idx]+1.5*K_Hmin_k_to_1[idx]*std_k_to_1[idx])
            coeff11 = []
            for aa in coeff_d_tot[idx]:
                if aa >=0:
                    coeff1 = pywt.threshold(coeff_d_tot[idx], Lambda_jH1,'hard')
                    coeff_d_tot[idx] = coeff1
                else:
                    coeff1 = pywt.threshold(coeff_d_tot[idx], Lambda_jL1,'hard')
                    coeff_d_tot[idx] = coeff1
        elif 0.01<val<0.2:
            Lambda_jL2 = (mean_k_to_1[idx]-(1-(Sj_L_k_to_1[idx])/(Sr_L))*K_Lmin_k_to_1[idx]*std_k_to_1[idx])
            Lambda_jH2 = (mean_k_to_1[idx]+(1-(Sj_H_k_to_1[idx])/(Sr_H))*K_Hmin_k_to_1[idx]*std_k_to_1[idx])
            for bb in coeff_d_tot[idx]:
                if bb >=0:
                    coeff2 = pywt.threshold(coeff_d_tot[idx], q*Lambda_jH2,'hard')
                    coeff_d_tot[idx] = coeff2
                else:
                    coeff2 = pywt.threshold(coeff_d_tot[idx], q*Lambda_jL2,'hard')
                    coeff_d_tot[idx] = coeff2

                    
    Approx_k_denoised = Approx_k[:]
    Approx_mean = np.mean(Approx_k[:])
    Approx_std = np.std(Approx_k[:])
    Approx_Kk_Lmin = (Approx_mean-np.abs(np.max(Approx_k[:])))/Approx_std
    Approx_Kk_Hmin = (np.abs(np.min(Approx_k[:]))-Approx_mean)/Approx_std

    Kj_L_approx = (1-(Sj_L_approx_rev[-k])/(Sr_L_cA))*Approx_Kk_Lmin
    Kj_H_approx = (1-(Sj_H_approx_rev[-k])/(Sr_H_cA))*Approx_Kk_Hmin

    Lambda_kL_approx = Approx_mean - Kj_L_approx*Approx_std
    Lambda_kH_approx = Approx_mean + Kj_H_approx*Approx_std

    for dd in range(len(Approx_k_denoised)):
        if dd>=0:
            coeffk_approx = pywt.threshold(Approx_k_denoised[dd],Lambda_kH_approx,'hard')
            Approx_k_denoised[dd] = coeffk_approx
        else:
            coeffk_approx = pywt.threshold(Approx_k_denoised[dd],Lambda_kL_approx,'hard')
            Approx_k_denoised[dd] = coeffk_approx


    coeff_total.append(Approx_k_denoised)
    for a in coeff_d_tot:
        coeff_total.append(a)
        
    coeff_denoised_all = coeff_total[:]

    denoised = pywt.waverec(coeff_denoised_all,wavelet)

    #RMSE of signal and denoised signal
    difference_array = np.subtract(x_ecg, denoised)
    squared_array = np.square(difference_array)
    root_mse = np.sqrt((squared_array.mean())/len(x_ecg))
    RMSE.append(root_mse)

    plt.ylim([-1.5,2.6])
    plt.axis("off")
    plt.plot(x_ecg, 'r')
    plt.savefig("signal_2.png")
    plt.clf()
    plt.ylim([-1.5,2.6])
    plt.axis("off")
    plt.plot(denoised, 'r')
    plt.savefig("denoised_2.png")


    #SSIM of signal and denoised signal
    image1 = Image.open("signal_2.png")
    image2 = Image.open("denoised_2.png")
    value = compare_ssim(image1, image2)
    print("SNR: {} Sigma: {} RMSE: {}".format(SNR, sigma, root_mse))
    SSIM.append(value)

    file=open("RMSE.txt", "a")
    file.write(str(root_mse))
    file.write("\n")
    file.close()

    file=open("SSIM.txt", "a")
    file.write(str(value))
    file.write("\n")
    file.close()

    #Plot 50Hz noise
##    plt.clf()
##    plt.axis("on")
##    plt.plot(time, data1, 'b', label="Nosiy Signal")
##    plt.plot(time, x_ecg, 'y', label="Original Signal")
##    plt.plot(time, denoised, 'r', label="Denoised Signal")
##    plt.legend()
##    plt.xlabel("Time / s")
##    plt.ylabel("Voltage / mV")
##    plt.show()
##

    plt.clf()
    plt.close()




print("\n")
rmse_mean=np.mean(RMSE)
ssim_mean=np.mean(SSIM)
rmse_std=np.std(RMSE)
ssim_std=np.std(SSIM)

print("Mean RMSE: {} STD of RMSE: {}".format(rmse_mean, rmse_std))
print("Mean SSIM: {} STD of SSIM: {}".format(ssim_mean, ssim_std))

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
ax1.plot(numbers, mean_rmse, 'ro', markersize=4)
ax1.tick_params(axis='y', labelcolor=color)
plt.errorbar(numbers, mean_rmse, yerr=rmse_error, ls='none', ecolor='black', elinewidth=1, capsize=1)
color='blue'
ax2 = ax1.twinx()  
ax2.set_ylabel('SSIM', color=color)  
ax2.plot(numbers, mean_ssim, 'bo', markersize=4)
ax2.tick_params(axis='y', labelcolor=color)
plt.errorbar(numbers, mean_ssim, yerr=ssim_error, ls='none', ecolor='black', elinewidth=1, capsize=1)
plt.title('Decimated \n ({} for {} Repeats)'.format(wavelet, repeats))

#plt.show()

    











