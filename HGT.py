#!/usr/bin/env python
# coding: utf-8

# <h2 style = "font-family: garamond; font-size: 24px; font-style: normal; letter-spcaing: 3px; background-color:#36609A ; color :#FFCE30 ; border-radius: 5px 5px; padding:10px;text-align:center; font-weight: bold" >Numerical Simulation of Horizontal Gene Transfer in A Competitive Lotka-Volterra Model </h2 > 
# <h2 style = "font-family: Times New Roman; font-size: 18px; font-style: normal; letter-spcaing: 3px; background-color:#36609A; color : #FFCE30 ; border-radius: 5px 5px; padding:3px;text-align:center; font-weight: bold" >Shiben Zhu,Teng Wang<br/>Feb 11,2023<br/>Email：shiben@link.cuhk.edu.hk</h2> 
# </div>

# In[31]:


#import modules
import time,scipy
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy import integrate
T1=time.perf_counter()#the code running time at the beginning

'''define different functions'''
# HGT basic equations
def HGT(t,p,NumSpecies,speed,fitness,effection,hgt,dilution,lost):
    equation=[]#empty list to collect the equations
    for i in list(range(NumSpecies)):
        for j in list(range(NumSpecies)):
            #creat three values to collect values
            #pai is continuous multiplication, set the initial value is a constant speed[i]
            #total is continuous summation,set the initial value is zero
            #sigma iscontinuous summation,set the initial value is zero
            pai,total,sigma=speed[i],0,0
            for k in list(range(NumSpecies)):
                pai*=(1+fitness[k]*p[i*NumSpecies+k]/p[i*NumSpecies+i])#'*=' means pai=pai*constant
                total+=effection[k,i]*p[k*NumSpecies+k]#'+=' means total=total+constant
                sigma+=p[k*NumSpecies+j]#'+=' means sigma=sigma+constant
                
            if i==j:
                eq1=pai*p[i*NumSpecies+i]*(1-total)-dilution*p[i*NumSpecies+i]#first equation
                equation.append(eq1) # add the first equation to the empty list one by one
            else:
                pai/=(1+fitness[j]*p[i*NumSpecies+j]/p[i*NumSpecies+i])#exclude the condition when k=j,'/=' means pai=pai/constant
                eq2=pai*(1+fitness[j])*p[i*NumSpecies+j]*(1-total)+hgt*sigma*(p[i*NumSpecies+i]-p[i*NumSpecies+j])-(dilution+lost)*p[i*NumSpecies+j]#the second equation
                equation.append(eq2) # add the second equation to the empty list one by one
            
    return equation

#randomize speed to get the matrix and caculate the probability by the matrix
def probability(threshod,Numspeed_rand,timespan,initial,NumSpecies,speed,effection,hgt,dilution,lost,positions):
    z=[]#empty list to collect the values
    
    for k in list(range(Numspeed_rand)):
        fitness=-0.1+0.2*np.random.rand(NumSpecies)#plasmid fitness values
        #get the answer of the equation
        sol=scipy.integrate.solve_ivp(HGT,
                                    t_span=timespan,
                                    y0=initial,
                                    method='RK45',
                                    args=(NumSpecies,speed,fitness,effection,hgt,dilution,lost),
                                    dense_output=True)
        s=sol.sol(timespan[1])[positions]
        z.append(np.min(s))# add the minum value to the empty list one by one
        
    '''calculate the probability!!!'''
    p=1-((np.round(z,decimals=len(str(threshod))-2).tolist().count(0))/Numspeed_rand)#the probability
    
    return p


# In[ ]:





# In[32]:


'''Competitive Lotka-Volterra equations'''
#initial values
threshod=0.00001#keep how many decimals
Numspeed_rand=1000#the times of randomizing speed
timespan=[0,2000]#time span
hgt=list(np.linspace(start=0,stop=0.4,num=100))#the level of Horizontal Gene Transfer
dilution=0.2#dilution value
lost=0.02#lost value
Possibility=pd.DataFrame()#creat an empty dataframe to collect and show the probability results

Num=list(range(2,40,1))##the number of Species

for NumSpecies in Num:
    positions=[x*NumSpecies+x for x in list(range(NumSpecies))]#find the positions when i=j which will help find the results in the loop
    media=np.random.rand(NumSpecies,NumSpecies)
    initial=((np.eye(NumSpecies,NumSpecies)*(media.max()-media.min())+media)/(media.max()-media.min()+1)).reshape(NumSpecies*NumSpecies)#initial Pij values
    speed=0.4*np.ones(NumSpecies)
    effection=0.8+0.1*np.random.rand(NumSpecies,NumSpecies)#interactive values
    for j in list(range(NumSpecies)):effection[j,j]=1

    # creat a loop to get the result
    for et,eta in enumerate(hgt):
        T3=time.perf_counter()#the code initial running time in the loop
        pos=probability(threshod,Numspeed_rand,timespan,initial,NumSpecies,speed,effection,eta,dilution,lost,positions)
        Possibility=pd.concat([Possibility,pd.DataFrame({'\u03B7':[eta],'Probability':[pos]})],axis=0)#creat a dataframe to save the results
        T4=time.perf_counter()#the code last running time in the loop
        print('The {num}th result(probability={probability}) needs {min}minutes in the loop'.format(num=et,
                                                                                                    probability=pos,
                                                                                                    min=round((T4-T3)/60)))#show the time once in the loop

    '''Save and show the results'''
    Possibility.reset_index(drop=True).to_excel('NumSpecies{NumSpecies}_Numspeed_rand{Numspeed_rand}.xlsx'.format(NumSpecies=NumSpecies,Numspeed_rand=Numspeed_rand))   

    plt.plot(Possibility[Possibility.columns[0]],Possibility[Possibility.columns[1]])#plot the curve
    plt.xlabel(Possibility.columns[0])#x-axis name
    plt.ylabel('Possibility')#y-axis name
    plt.title('Mulitiple Lotka-Volterra System with {label}'.format(label=Possibility.columns[0]))#figure name
    plt.savefig('NumSpecies{NumSpecies}_Numspeed_rand{Numspeed_rand}.png'.format(NumSpecies=NumSpecies,Numspeed_rand=Numspeed_rand))
    print('the figure of Mulitiple Lotka-Volterra System with {label} is successfully plotted.'.format(label=Possibility.columns[0]))
    T2=time.perf_counter()#the code running time at the end
    print('Total Time Consumption of The Model Codes:%s minutes'%(round((T2-T1)/60)))


# <h2 style = "font-family: Times New Roman; font-size: 18px; font-style: normal; letter-spcaing: 3px; background-color:#36609A; color : #FFCE30 ; border-radius: 5px 5px; padding:3px;text-align:center; font-weight: bold" >------------------------------------The End------------------------------------</h2> 
# </div>
