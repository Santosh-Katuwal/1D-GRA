import os
import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

H=540 #ft
VSs=1500 #ft/sec
VSr=5000
ro_s=125 #lb/ft^3
ro_r=160
jai_s=0.05 #jai=5%
jai_r=0.02

#reading current working directory
cwd=os.getcwd()

#Reading data from csv file
t=pd.read_csv(cwd+'\\EW.csv',usecols=[0])
a=pd.read_csv(cwd+'\\EW.csv',usecols=[1])/981

n=len(a)

#computing fft using numpy package
acc=a.values.flatten()
dt=float(t.iloc[50]-t.iloc[49])

fft= rfft(acc)*dt       #fft is complex valued variable
freq=rfftfreq(n, d=dt)

#computing absolute fft. 
FFT=abs(fft)

#Calculating complex impedance ratio (alpha*)
num=ro_s*VSs*(1+1j*jai_s)
den=ro_r*VSr*(1+1j*jai_r)
alpha_z=num/den

#**************************************************************************
#calculating Transfer function for first half part
#**************************************************************************
TF=[]   #defining null variable to store TF data from loop
abs_TF=[]   #defining null variable to store absolute TF data
for i in range(0,len(freq)):
    omega=2*np.pi*freq[i]
    V_star=VSs*(1+1j*jai_s)
    ks=omega/V_star
    KH=ks*H
    
    TFi=1/(np.cos(KH)+1j*np.sin(KH)*alpha_z)    #computed TF for each loop entry
    TF.append(TFi)  #Collecting transfer function in TF variable
    
    abs_TFi=abs(TFi)    #Computing absolute value of transfer function
    abs_TF.append(abs_TFi)  #capturing data from loop entry

#**************************************************************************
#calculating Transfer function for second half part
#**************************************************************************    
#to calculate transfer function of remaining half part= conjugate of previous half with reverse order
#e.g. first half: 1+2j,1+3j,1+4j,1+5j,1+6j
#second half: 1-6j, 1-5j, 1-4j, 1-3j, 1-2j
rev_TF=[]   
rev_con_TF=[]
for i in range(len(freq),1,-1):
    rev_TFi=TF[i-1]         #Reversing TF of first half
    rev_TF.append(rev_TFi)  #Capturing reversed data from loop
    rev_con_TFi=np.conj(rev_TFi)    #Conjugating reversed TF
    rev_con_TF.append(rev_con_TFi)  #capturing conjugated Transfer function

TF1=pd.DataFrame(TF)
TF2=pd.DataFrame(rev_con_TF)

all_TF=TF1.append(TF2) #full scale transfer function
full_TF=all_TF.values.flatten() #converting TF to array to make callable
#************************************************************************
#computing full Scale fourier spectra of input motion
#************************************************************************
a_fft=np.array(np.fft.fft(a, axis=0))/n
abs_a_fft=abs(a_fft)
#computing single sided fourier amplitude spectra
FFT=[]
for m in range(0,len(freq)):
    FFTi=2*abs_a_fft[m]
    FFT.append(FFTi)
    

#************************************************************************
#New FFT= full_scale_fft* Full_scale_TF
#************************************************************************
new_fft=[]
for i in range(0,n):
    new_ffti=a_fft[i]*full_TF[i]
    new_fft.append(new_ffti)
    i=i+1

#************************************************************************
#Fourier amplitude of Output motion
#************************************************************************
out_fft=[]
out_absfft=[]
for i in range(0,len(freq)):
    out_ffti=2*new_fft[i]
    out_fft.append(out_ffti)
    
    out_absffti=abs(out_ffti)
    out_absfft.append(out_absffti)
    

#************************************************************************
#output acceleration
#************************************************************************
t=np.array(t)
a=np.array(a)
out_acc=np.real(np.array(np.fft.ifft(new_fft, axis=0)))*n
a_out=np.array(out_acc)

img=plt.imread("layer.png")

fig, [ax1, ax2, ax3] = plt.subplots(3, 1,figsize=(15,9))
N=len(t)
maxx=max(a)
minn=min(a)

ax1 = plt.subplot(3,1,1)
ax2=plt.subplot(3,1,2)
ax3=plt.subplot(3,1,3)
#'''
for i in range(N+1):
    if i<N-1:
        ax1.plot([t[i],t[i+1]],[a_out[i],a_out[i+1]],lw=1,color='r')
        ax1.set_xlim(0,max(t))
        ax1.set_ylim(minn*1.5,maxx*1.5)
        ax1.set_xlabel('time(sec)')
        ax1.set_ylabel('output: acc[g]')
        plt.tight_layout()
        
        ax2.cla()
        img=plt.imread("layer.png")
        ax2.text(1000, 300, 'Soil Layer', style='italic',fontsize=20)
        ax2.text(2500, 100, 'Output surface motion', style='italic',fontsize=12,color='r')
        ax2.text(1000, 500, 'Rock Layer', style='italic',fontsize=20)
        ax2.text(2500, 350, 'Input bed rock motion', style='italic',fontsize=12)
        ax2.imshow(img)
        ax2.plot(a_out[i]*200+2500,160,marker='o',markersize=15,color='r') #amplitude scaled by 100 times
        ax2.plot(a[i]*200+2500,400,marker='o',markersize=15,color='k')
        ax2.set_xticks([])
        ax2.set_yticks([])
        #ax.plot([0,trans[i]], [2,10],color='k',lw=5)
        #ax.set_xlim([minn*3,maxx*3])
        #ax.set_ylim([0,12])
        #ax.set_xlabel('Amplitude')
        #ax.set_yticks([])
        #plt.tight_layout()
        
        ax3.plot([t[i],t[i+1]],[a[i],a[i+1]],lw=1,color='k')
        ax3.set_xlim(0,max(t))
        ax3.set_ylim(minn*1.2,maxx*1.2)
        ax1.set_xlabel('time(sec)')
        ax1.set_ylabel('input: acc[g]')
        plt.tight_layout()
        plt.pause(0.05)
#'''
'''
plt.subplot(5,1,1)
plt.plot(t,a,lw=1,c='k')
plt.xlabel('Time(sec)')
plt.ylabel('acc[g]')
plt.subplot(5,1,2)
plt.plot(freq,FFT,lw=1,c='k')
plt.xlabel('Freq(Hz)')
plt.ylabel('Fourier Amplitude')
plt.subplot(5,1,3)
plt.plot(freq,abs_TF,lw=1,c='k')
plt.xlabel('Freq(Hz)')
plt.ylabel('|F3|')
plt.subplot(5,1,4)
plt.plot(freq,out_absfft,lw=1,c='k')
plt.xlabel('Freq(Hz)')
plt.ylabel('Fourier Amplitude')
plt.subplot(5,1,5)
plt.plot(t,out_acc,lw=1,c='k')
plt.xlabel('Time(sec)')
plt.ylabel('acc[g]')
plt.tight_layout()
'''



