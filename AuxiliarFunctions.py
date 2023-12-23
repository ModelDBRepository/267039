import numpy as np
from scipy import integrate
from scipy import signal
from scipy.signal import kaiserord, lfilter, firwin
from scipy.signal import find_peaks


"""
This code describes the auxiliar functions used in the following paper:

!!!!!!


If you use this code, please cite:
 
!!!!!!!!

"""



################################################################################
#External drive time dependence
################################################################################

def linear_increasing_vector(vo,vf,duration,time_step):
    number_steps=int(duration/time_step)
    inclination=float(vf-vo)/number_steps
    vector=np.arange(number_steps)*inclination + vo

    return vector


def IrregularFluctuationPattern(v_base,v_new,TransitionTime,MinPlatoTime,MaxPlatoTime,BaseTime,DT,T_simulation):

    v_low=v_base; v_high=v_new
    ramp_up=linear_increasing_vector(v_low,v_high,TransitionTime,DT)
    ramp_down=linear_increasing_vector(v_high,v_low,TransitionTime,DT)
    plato_base=np.full(int(BaseTime/DT),v_base,dtype=float)
    
    Pattern=np.asarray([],dtype=float) ; STOP=0
    #-----------------------------------------------------------------------------------
    while(STOP==0):

        PlatoTime=np.random.uniform(low=MinPlatoTime,high=MaxPlatoTime,size=1)  
        plato=np.full(int(PlatoTime/DT),v_new,dtype=float)
       
        Pattern_cicle=np.r_[np.r_[np.r_[plato_base,ramp_up],plato],ramp_down]

        Pattern=np.r_[Pattern,Pattern_cicle]

        PatternLen=len(Pattern)

        if(PatternLen>=(int(T_simulation/DT)-len(plato_base))):
            STOP=1
            if(PatternLen>int(T_simulation/DT)):
                Pattern_final=Pattern[:int(T_simulation/DT)]
            else:
                Pattern_final=np.r_[Pattern,np.full(int(T_simulation/DT)-len(Pattern),v_low,dtype=float)]

    #-------------------------------------------------------------------------------------

    Pattern_time=np.arange(0,T_simulation,DT)
    
    return Pattern_time, Pattern_final




#######################################################
#Z-Score normalization in a Signal
#######################################################
#Signal: np.array

def ZscoreNorm(Signal):

    Signal_normalized=(Signal-np.mean(Signal))/np.std(Signal)

    return  Signal_normalized



####################################################################
#Defining Band-Filter function to Hilbert Transform
####################################################################

#------Filter Parameters-----------
#width = 5.0 #The desired width in Hz of the transition from pass to stop
#ripple_db = 60.0 # The desired attenuation in the stop band, in dB.
#lowcut=30. ; highcut=50. #Hz 

def Keiser_bandpass(data, time, lowcut, highcut, ripple_db, width, Nyquist,SamplingRate):
    numtaps, beta = kaiserord(ripple_db, width/Nyquist)
    if(numtaps%2==0):
        numtaps=(numtaps+1) #Numtaps must be odd
    taps = firwin(numtaps,[lowcut/Nyquist, highcut/Nyquist], window=('kaiser', beta), nyq=1.0, pass_zero=False, scale=True) 
    filtered_data = lfilter(taps, 1.0, data)
    delay = 0.5 * (numtaps-1) /SamplingRate #To corrected to zero-phase
    delay_index=int(np.floor(delay*SamplingRate))
    filtered_data= filtered_data[numtaps-1:]#taking out the "corrupted" signal
    time_filteredData=time[numtaps-1:]-delay #correcting the delay and taking out the "corrupted" signal part
    data_thatOverLaps_FileteredOne= data[(numtaps-1-delay_index):(len(data)-(numtaps-1-delay_index))]


    return time_filteredData, filtered_data, data_thatOverLaps_FileteredOne



######################################################################################################
#Phase associated to a signal through the Hilbert transform:
######################################################################################################

#SamplingSpace: in seconds (no units)
#Signal_time: in seconds (no units)
#lowcut: in Hz (no units)(band to filter)
#highcut: in Hz (no units)(band to filter)


def SignalPhase_byHilbert(Signal, Signal_time, lowcut, highcut,SamplingSpace):

    Signal = Signal-np.mean(Signal)

    width = 5.0 #The desired width in Hz of the transition from pass to stop
    ripple_db = 60.0 # The desired attenuation in the stop band, in dB.
    SamplingRate=1./SamplingSpace
    Nyquist=SamplingRate/2.

    time_Filtered_Signal, Filtered_Signal, Signal_thatOverLaps_FileteredOne= Keiser_bandpass(Signal, Signal_time, lowcut, highcut, ripple_db, width, Nyquist,SamplingRate)#Use everything in seconds to obtain in seconds

    #--------------------------------------------------------------------------
    #The hilbert transform are very slow when the signal has odd lenght,
    #This part check if the the lenght is odd, and if this is the case it adds a zero in the end
    #of all the vectors related to the filtered Signal:

    if(len(Filtered_Signal)%2!=0):#If the lengh is odd
        tmp1=Filtered_Signal.tolist(); tmp1.append(0)
        tmp2=time_Filtered_Signal.tolist(); tmp2.append((len(time_Filtered_Signal)+1)*SamplingSpace + time_Filtered_Signal[0])
        tmp3=Signal_thatOverLaps_FileteredOne.tolist(); tmp3.append(0)
        Filtered_Signal=np.asarray(tmp1); time_Filtered_Signal=np.asarray(tmp2)
        Signal_thatOverLaps_FileteredOne=np.asarray(tmp3)
    #--------------------------------------------------------------------------
    
    HT_Filtered_Signal = signal.hilbert(Filtered_Signal); Envelope=np.abs(HT_Filtered_Signal)
    Phase=np.angle(HT_Filtered_Signal)#The phase is between -pi and pi in radians
   
    return time_Filtered_Signal,Filtered_Signal,Signal_thatOverLaps_FileteredOne,Envelope,Phase




