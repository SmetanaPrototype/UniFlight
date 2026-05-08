import matplotlib.pyplot as plt
import math as m
import rocket_parser as rp
import basis
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate

# Lists init
numeric = []
length = []
asc_length = []
mass = []
stiffness = []

# Lambda oscillations constants
L = basis.lamb

# Local functions
def calculate_sum(base):
    summ = [0] * len(base)
    for i in range(len(base)):
        if i == 0:
            summ[i] = base[i]
        else:
            summ[i] = summ[i-1] + base[i - 1] + base[i]
    return summ

parser = rp.rocket_parser()
work_time = parser.get_work_time()

numeric   =  parser.numeric
stiffness =  parser.stiffnesses
length    =  parser.steps
masses    =  parser.masses
alength   =  parser.asc_length

block_number    = int(parser.get_block_number())
delta_mass_fu   = parser.get_delta_mass_fu()
sector_range_fu = parser.get_coordinates_fu()
delta_mass_ox   = parser.get_delta_mass_ox()
sector_range_ox = parser.get_coordinates_ox()

ti = 0.0
ver_mass_vector = []
time_vector = []
freq_vector_1 = []
freq_vector_2 = []
freq_vector_3 = []
freqmass_vector_1 = []
freqmass_vector_2 = []
freqmass_vector_3 = []
total_iterations = 0

time_vector = [t for t in range(0, 80+1, basis.timestep)]
ver_mass_vector = [parser.changed_mass(ti) for ti in time_vector]
total_iterations = len(ver_mass_vector)

start_color = [0.68, 0.85, 0.9]
end_color = [0, 0, 0.55]

custom_lines = [
   Line2D([0], [0], color='blue', lw=2),
   Line2D([0], [0], color='red', lw=2),
   Line2D([0], [0], color='black', lw=2),
   Line2D([0], [0], color='green', lw=2),
   Line2D([0], [0], color='yellow', lw=2)
]

plt.figure(figsize=(8, 8))

def output(parser):
   """Save results to file"""
   rocketname = parser.name
   basis.write_arrays_to_csv("output/"+rocketname+"_oscillations.csv",
                                length   =numeric,
                                form_1     =f_stiffness[0],
                                form_2     =f_stiffness[1],
                                form_3     =f_stiffness[2],
                                difform_1  =f_stiffness_diff[0],
                                difform_2  =f_stiffness_diff[1],
                                difform_3  =f_stiffness_diff[2],
                                intform_1  =f_stiffness_integral[0],
                                intform_2  =f_stiffness_integral[1],
                                intform_3  =f_stiffness_integral[2])

   basis.write_arrays_to_csv("output/"+rocketname+"_frequency.csv",
                                time       =time_vector,
                                freq_1     =freq_vector_1,
                                freq_2     =freq_vector_2,
                                freq_3     =freq_vector_3,
                                freq_mass_1=freqmass_vector_1,
                                freq_mass_2=freqmass_vector_2,
                                freq_mass_3=freqmass_vector_3)


for mo, mass in enumerate(ver_mass_vector):

   print(round(mo/len(ver_mass_vector)*100), "%", end="\r")
   def delta_vector(previous, actual):
    f_12 = [x ** 2 for x in previous]
    mf_12 = basis.calculate_multi(f_12, mass)
    sum_mf_12 = calculate_sum(mf_12)
    f1_f20 = basis.calculate_multi(previous, actual)
    f1_f20_mass = basis.calculate_multi(f1_f20, mass)
    f1_f20_mass_summ = calculate_sum(f1_f20_mass)
    delta12 = - f1_f20_mass_summ[-1]/sum_mf_12[-1]
    delta12f = [x * delta12 for x in previous]
    return delta12f

   m_N = basis.calculate_multi(numeric, mass)

   sum_m = calculate_sum(mass)
   sum_m_N = calculate_sum(m_N)

   Nmid = sum_m_N[-1]/sum_m[-1]

   N_Nm  = [x - Nmid for x in numeric]
   N_Nm2 = [x ** 2 for x in N_Nm]

   preIn = basis.calculate_multi(N_Nm2, mass)
   In = calculate_sum(preIn)

   rocket_length = parser.rocket_length
   rocket_mass = parser.full_mass

   a = [x/rocket_length for x in L]
   chJ_cosJ = [m.cosh(x)-m.cos(x) for x in L]
   shJ_sinJ = [m.sinh(x)-m.sin(x) for x in L]
   sinJ_shJ = [m.sin(x)-m.sinh(x) for x in L]
   Y = [a / b for a, b in zip(chJ_cosJ, sinJ_shJ)]


   f_zero = [0] * basis.mode_num
   f_stiffness = [0] * basis.mode_num
   f_stiffness_diff = [0] * basis.mode_num
   f_stiffness_integral = [0] * basis.mode_num
   f_mass = [0] * basis.mode_num
   w_calc = [0] * basis.mode_num
   fi     = [0] * basis.mode_num
   mass_s = [0] * basis.mode_num
   for i in range(len(f_zero)):
       f_zero[i] = list(((m.sin(a[i]*x)+m.sinh(a[i]*x))*Y[i]+(m.cos(a[i]*x)+m.cosh(a[i]*x)))/2 for x in numeric)

   w_zero = [m.sqrt(max(stiffness)/(rocket_mass*(10**3)/rocket_length*pow(rocket_length,4)))*(x**2)/(2*m.pi) for x in L]

   def calculate_form(index, time_idx):

       f_start = f_zero[index]
       tolerance = 1e-8
       mass_effective = parser.effective_mass(time_idx, index)
       iteration = 0
       while(True):
           m_f1 = basis.calculate_multi(mass_effective, f_start)
           sum_m_f1 = calculate_sum(m_f1)

           value_6_11 = basis.calculate_multi(m_f1, N_Nm)
           sum_value_6_11 = calculate_sum(value_6_11)

           D1 = - sum_value_6_11[-1]/In[-1]
           D2 = - sum_m_f1[-1]/sum_m[-1]

           D1_6 = [x*D1 for x in N_Nm]
           D2_15 = [D2+x  for x in D1_6]
           accumulated_delta = [0] * len(f_start)
           if index > 0:
               for i in range(index):
                   newin = delta_vector(f_mass[index - 1 - i], f_start)
                   accumulated_delta = [a  + b  for a, b in zip(accumulated_delta, newin)]

           f1_16 = [a + b + c for a, b, c in zip(D2_15, f_start, accumulated_delta)]
           f_mass[index] = [x/max(f1_16) for x in f1_16]

           m_f1 = basis.calculate_multi(mass_effective, f_mass[index])
           sum_m_f1 = calculate_sum(m_f1)
           double_sum_m_f1 = calculate_sum(sum_m_f1)
           dm1 = [-x*double_sum_m_f1[-1]/numeric[-1] for x in numeric]
           M1x = [a + b for a, b in zip(dm1, double_sum_m_f1)]
           M1x_E = [a/b if b != 0 else 0 for a, b in zip(M1x, stiffness)]
           sum_M1x_E = calculate_sum(M1x_E)
           fi[index] = calculate_sum(sum_M1x_E)
           double_sum_M1x_E_mass = basis.calculate_multi(fi[index], mass_effective)
           summ_13 = calculate_sum(double_sum_M1x_E_mass)

           value_13_15 = basis.calculate_multi(double_sum_M1x_E_mass, N_Nm)
           sum_13_15 = calculate_sum(value_13_15)

           D1 = - sum_13_15[-1]/In[-1]
           D2 = - summ_13[-1]/sum_m[-1]

           D1_15 = [x*D1 for x in N_Nm]

           D2_11 = [a + b + D2 for a, b in zip(fi[index], D1_15)]
           accumulated_delta = [0] * len(fi[index])
           if index > 0:
               for i in range(index):
                   newin = delta_vector(f_stiffness[index - 1 - i], fi[index])
                   accumulated_delta = [(a + b) for a, b in zip(accumulated_delta, newin)]

           D2_11 = [a + b  for a, b in zip(D2_11, accumulated_delta)]

           f_stiffness_res = [x/max(D2_11,key=abs) for x in D2_11]


           m_f1 = basis.calculate_multi(mass_effective, f_stiffness_res)
           sum_m_f1 = calculate_sum(m_f1)
           double_sum_m_f1 = calculate_sum(sum_m_f1)
           dm1 = [-x*double_sum_m_f1[-1]/numeric[-1] for x in numeric]
           M1x = [a + b for a, b in zip(dm1, double_sum_m_f1)]
           M1x2 = [x ** 2 for x in M1x]
           M1x2_E = [a/b if b != 0 else 0 for a, b in zip(M1x2, stiffness)]
           sum_M1x2_E = calculate_sum(M1x2_E)
           f_12 = [x ** 2 for x in f_stiffness_res]
           mf_12 = basis.calculate_multi(f_12, mass_effective)
           sum_mf_12 = calculate_sum(mf_12)
           w_calc[index] = m.sqrt(sum_mf_12[-1]/(sum_M1x2_E[-1]*1000.0*pow(length[-1]/2,4)))/(2*m.pi)

           mass_s[index] = sum_mf_12[-1]
           if max(abs(f_stiffness_res[i] - f_start[i]) for i in range(len(f_start))) < tolerance:
                break
           if iteration >= 300:
                break
           f_start = f_stiffness_res
           iteration+=1

       return f_stiffness_res
   #################################################################

   w_femap = [11.86, 32.51, 60.66, 86.75, 124.37]
   color_pairs = [
       ([0.68, 0.85, 0.9], [0, 0, 1]),
       ([1, 0.68, 0.68], [1, 0, 0]),
       ([0.8, 0.8, 0.8], [0, 0, 0])
   ]

   for i in range(basis.mode_num):
       f_stiffness[i] = calculate_form(i, mo*basis.timestep)
       dif_y = np.diff(np.array(f_stiffness[i]))
       dif_x = np.diff(np.array(alength))
       f_stiffness_diff[i] = dif_y/dif_x
       f_stiffness_diff[i] = np.insert(f_stiffness_diff[i], 0, f_stiffness_diff[i][0])
       f_stiffness_integral[i] = [integrate.trapezoid(f_stiffness[i], alength)] *  len(f_stiffness[i])
       if (len(f_stiffness_diff[i])!=len(f_stiffness[i])):
           print("original and differential oscill forms are not synchronized")
           exit()

       plt.subplot(2, 1, 1)
       plt.plot(alength, f_stiffness[i], color = basis.interpolate_color(color_pairs[i][0], color_pairs[i][1], mo, total_iterations))
       if mo==0:
           plt.plot(alength, f_stiffness[i], color = 'g', linewidth=5)
       if mo == len(ver_mass_vector) - 1:
           plt.plot(alength, f_stiffness[i], color = 'y')

       plt.title('Расчет форм колебаний', fontsize=16)
       plt.xlabel('Длина РН, м', fontsize=14)
       plt.ylabel('Форма', fontsize=14)
       plt.grid(True)
       plt.tight_layout()
       plt.legend(custom_lines, ['1 Тон', '2 Тон', '3 Тон', '0 секунда', '130 секунда'])

       plt.subplot(2, 1, 2)
       plt.plot(alength, f_stiffness_diff[i], color = basis.interpolate_color(color_pairs[i][0], color_pairs[i][1], mo, total_iterations))
       if mo==0:
           plt.plot(alength, f_stiffness_diff[i], color = 'g', linewidth=5)
       if mo == len(ver_mass_vector) - 1:
           plt.plot(alength, f_stiffness_diff[i], color = 'y')
       plt.title('Расчет производных форм колебаний', fontsize=16)
       plt.xlabel('Длина РН, м', fontsize=14)
       plt.ylabel('Производная формы', fontsize=14)
       plt.grid(True)
       plt.tight_layout()
       plt.legend(custom_lines, ['1 Тон', '2 Тон', '3 Тон', '0 секунда', '130 секунда'])

   freq_vector_1.append(w_calc[0])
   freq_vector_2.append(w_calc[1])
   freq_vector_3.append(w_calc[2])
   freqmass_vector_1.append(mass_s[0])
   freqmass_vector_2.append(mass_s[1])
   freqmass_vector_3.append(mass_s[2])
   #print(mass)

output(parser)

plt.show()

plt.subplot(2, 1, 1)
plt.title('Расчет частот колебаний', fontsize=16)
plt.xlabel('Время полета, с', fontsize=14)
plt.ylabel('Частота, Гц', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.plot(time_vector, freq_vector_1,color='blue', label = '1 Тон')
plt.plot(time_vector, freq_vector_2,color='red', label = '2 Тон')
plt.plot(time_vector, freq_vector_3,color='black', label = '3 Тон')
plt.legend()
plt.subplot(2, 1, 2)
plt.title('Расчет приведенной массы', fontsize=16)
plt.xlabel('Время полета, с', fontsize=14)
plt.ylabel('Приведенная масса, кг', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.plot(time_vector, freqmass_vector_1,color='blue', label = '1 Тон')
plt.plot(time_vector, freqmass_vector_2,color='red', label = '2 Тон')
plt.plot(time_vector, freqmass_vector_3,color='black', label = '3 Тон')
plt.legend()
plt.show()