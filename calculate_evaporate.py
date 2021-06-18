import math
import pandas as pd
import sympy

class Constant:
    # density
    rho_H2O = 997 # [kg/m^3]
    rho_PG = 1036 # [kg/m^3]
    
    # gas constant
    R = 8.314 # [J/(mol·K)]

    # constants in Antonie Eq.
    A_PG = 7.66
    C_PG = 160.94
    B_PG = 1668.61

    A_H2O = 8.02754
    C_H2O = 231.405
    B_H2O = 1705.616

    # mole weight
    M_PG = 0.07609 # [kg/mol]
    M_H2O = 0.018015 # [kg/mol]
    M_air = 0.0289644 # [kg/mol]

    # constants in Fuller Eq.
    PG_sigma_vi = 78.4
    H2O_sigma_vi = 13.1
    sigma_vair = 19.7


class HuLarsonEq(object):
    '''
    Input:
        theta: contact angle, [°]
        R: bottom radius, [m]
        Mj: volume, [pL]
        T_C: input temperature, [℃]
        T_K: input temperature, [K]
        RH_H2O: retative humidity of water vapor in the environment, [%], ex. RH = 75
        RH_PG: retative humidity of PG vapor in the environment, [%]
        wt_frac_PG: weight fraction of PG, pure water wt_frac_PG = 0, pure PG wt_frac_PG = 1
        Patm: environment pressure [atm], default is 1

    Output:
        D: gas diffusion coefficient, [m^2/s]
        Cv: vapor concentration of droplet surface, [kg/m^3]
        Ca: vapor concentration of atmosphere, [kg/m^3]
        J: evaporation rate, [kg/(m^2·s)]
    '''
    
    def __init__(self, theta, bottom_radius, Mj, T_C, RH, wt_frac_PG, Patm=1):
        self.theta = theta
        self.R = bottom_radius
        self.Mj = Mj
        self.T_C = T_C
        self.T_K = T_C + 273.15
        self.RH_H2O = RH
        self.RH_PG = 0
        self.wt_frac_PG = wt_frac_PG
        self.Patm = Patm

        self.T_env = self.T_K
        self.T_ink = self.T_K

    def diffusion_coefficient(self):
        if self.wt_frac_PG == 0:
            K1_H2O = math.sqrt((Constant.M_H2O + Constant.M_air) / (Constant.M_H2O * Constant.M_air) / 1000)
            K2_H2O = (Constant.H2O_sigma_vi ** (1 / 3) + Constant.sigma_vair ** (1 / 3)) ** 2
            self.D_H2O = 10 ** (-7) * self.T_K ** 1.75 * K1_H2O / (self.Patm * K2_H2O)
        elif self.wt_frac_PG == 1:
            K1_PG = math.sqrt((Constant.M_PG + Constant.M_air) / (Constant.M_PG * Constant.M_air) / 1000)
            K2_PG = (Constant.PG_sigma_vi ** (1 / 3) + Constant.sigma_vair ** (1 / 3)) ** 2
            self.D_PG = 10 ** (-7) * self.T_K ** 1.75 * K1_PG / (self.Patm * K2_PG)
        else:
            K1_PG = math.sqrt((Constant.M_PG + Constant.M_air) / (Constant.M_PG * Constant.M_air) / 1000)
            K2_PG = (Constant.PG_sigma_vi ** (1 / 3) + Constant.sigma_vair ** (1 / 3)) ** 2
            K1_H2O = math.sqrt((Constant.M_H2O + Constant.M_air) / (Constant.M_H2O * Constant.M_air) / 1000)
            K2_H2O = (Constant.H2O_sigma_vi ** (1 / 3) + Constant.sigma_vair ** (1 / 3)) ** 2
            self.D_PG = 10 ** (-7) * self.T_K ** 1.75 * K1_PG / (self.Patm * K2_PG)
            self.D_H2O = 10 ** (-7) * self.T_K ** 1.75 * K1_H2O / (self.Patm * K2_H2O)

    def vapor_density_surface(self):
        mole_frac_PG = wt2mole_frac(self.wt_frac_PG)
        mole_frac_H2O = 1 - mole_frac_PG
        if self.wt_frac_PG == 0:
            Pi_star_H2O = math.pow(10, (Constant.A_H2O - (Constant.B_H2O / (Constant.C_H2O + self.T_ink - 273.15))))
            Pi_star_H2O = Pi_star_H2O * 101325 / 760    
            self.Cv_H2O = Constant.M_H2O * Pi_star_H2O * mole_frac_H2O / Constant.R / self.T_ink
        elif self.wt_frac_PG == 1:
            Pi_star_PG = math.pow(10, (Constant.A_PG - (Constant.B_PG / (Constant.C_PG + self.T_ink - 273.15))))
            Pi_star_PG = Pi_star_PG * 101325 / 760
            self.Cv_PG = Constant.M_PG * Pi_star_PG * mole_frac_PG / Constant.R / self.T_ink        
        else:
            Pi_star_PG = math.pow(10, (Constant.A_PG - (Constant.B_PG / (Constant.C_PG + self.T_ink - 273.15))))
            Pi_star_PG = Pi_star_PG * 101325 / 760
            Pi_star_H2O = math.pow(10, (Constant.A_H2O - (Constant.B_H2O / (Constant.C_H2O + self.T_ink - 273.15))))
            Pi_star_H2O = Pi_star_H2O * 101325 / 760
            self.Cv_PG = Constant.M_PG * Pi_star_PG * mole_frac_PG / Constant.R / self.T_ink
            self.Cv_H2O = Constant.M_H2O * Pi_star_H2O * mole_frac_H2O / Constant.R / self.T_ink

    def vapor_density_environment(self):
        if self.wt_frac_PG == 0:
            Pei_star_H2O = math.pow(10, (Constant.A_H2O - (Constant.B_H2O / (Constant.C_H2O + self.T_env - 273.15))))
            Pei_star_H2O = Pei_star_H2O * 101325 / 760 
            self.Ca_H2O = Constant.M_H2O * Pei_star_H2O * (self.RH_H2O / 100) / Constant.R / self.T_env
        elif self.wt_frac_PG == 1:
            Pei_star_PG = math.pow(10, (Constant.A_PG - (Constant.B_PG / (Constant.C_PG + self.T_env - 273.15))))
            Pei_star_PG = Pei_star_PG * 101325 / 760
            self.Ca_PG = Constant.M_PG * Pei_star_PG * (self.RH_PG / 100) / Constant.R / self.T_env
        else:
            Pei_star_PG = math.pow(10, (Constant.A_PG - (Constant.B_PG / (Constant.C_PG + self.T_env - 273.15))))
            Pei_star_PG = Pei_star_PG * 101325 / 760
            Pei_star_H2O = math.pow(10, (Constant.A_H2O - (Constant.B_H2O / (Constant.C_H2O + self.T_env - 273.15))))
            Pei_star_H2O = Pei_star_H2O * 101325 / 760 
            self.Ca_PG = Constant.M_PG * Pei_star_PG * (self.RH_PG / 100) / Constant.R / self.T_env
            self.Ca_H2O = Constant.M_H2O * Pei_star_H2O * (self.RH_H2O / 100) / Constant.R / self.T_env
    
    def evaporation_rate(self):
        self.R = self.R * 10 ** (-6)
        if self.wt_frac_PG == 0:
            self.J_H2O = (self.D_H2O * (self.Cv_H2O - self.Ca_H2O) / self.R * (math.sin(math.radians(self.theta))) ** 2 * 
                        (0.135 * math.radians(self.theta) ** 2 + 0.65) / (1 - math.cos(math.radians(self.theta))))
            self.J = self.J_H2O
        elif self.wt_frac_PG == 1:
            self.J_PG = (self.D_PG * (self.Cv_PG - self.Ca_PG) / self.R * (math.sin(math.radians(self.theta))) ** 2 * 
                        (0.135 * math.radians(self.theta) ** 2 + 0.65) / (1 - math.cos(math.radians(self.theta))))
            self.J = self.J_PG
        else:
            self.J_PG = (self.D_PG * (self.Cv_PG - self.Ca_PG) / self.R * (math.sin(math.radians(self.theta))) ** 2 * 
                        (0.135 * math.radians(self.theta) ** 2 + 0.65) / (1 - math.cos(math.radians(self.theta))))

            self.J_H2O = (self.D_H2O * (self.Cv_H2O - self.Ca_H2O) / self.R * (math.sin(math.radians(self.theta))) ** 2 * 
                        (0.135 * math.radians(self.theta) ** 2 + 0.65) / (1 - math.cos(math.radians(self.theta))))
            self.J = self.J_H2O + self.J_PG
    
    def get_weight(self):
        rho_mixture = get_mixture_density(wt2mole_frac(self.wt_frac_PG), self.T_K)
        wt_total = self.Mj * 10 ** (-15) * rho_mixture
        wt_PG =  wt_total * self.wt_frac_PG
        wt_H2O = wt_total * (1 - self.wt_frac_PG)
        if self.wt_frac_PG == 0:
            return {
                'wt_frac_PG': get_wt_frac(wt_PG, wt_H2O), 
                'wt_H2O': wt_H2O, 
                } 
        elif self.wt_frac_PG == 1:
            return {
                'wt_frac_PG': get_wt_frac(wt_PG, wt_H2O), 
                'wt_PG': wt_PG
                } 
        else:
            return {
                'wt_frac_PG': get_wt_frac(wt_PG, wt_H2O), 
                'wt_H2O': wt_H2O, 
                'wt_PG': wt_PG
                } 

    def calculate(self):
        self.diffusion_coefficient()
        self.vapor_density_surface()
        self.vapor_density_environment()
        self.evaporation_rate()
        if self.wt_frac_PG == 0:
            return {
                'D_H2O': self.D_H2O,
                'Cv_H20': self.Cv_H2O,
                'Ca_H2O': self.Ca_H2O,
                'J_H2O': self.J_H2O,
                'J': self.J 
                } 
        elif self.wt_frac_PG == 1:
            return {
                'D_PG': self.D_PG, 
                'Cv_PG': self.Cv_PG,
                'Ca_PG': self.Ca_PG,
                'J_PG': self.J_PG,
                'J': self.J 
                } 
        else:
            return {
                'D_PG': self.D_PG, 
                'D_H2O': self.D_H2O,
                'Cv_PG': self.Cv_PG,
                'Cv_H20': self.Cv_H2O,
                'Ca_PG': self.Ca_PG,
                'Ca_H2O': self.Ca_H2O,
                'J_PG': self.J_PG,
                'J_H2O': self.J_H2O,
                'J': self.J 
                } 


def wt2mole_frac(wt_frac_PG):
    '''
    change weight fraction of PG to mole fraction of PG
    '''
    mole_PG = wt_frac_PG / Constant.M_PG
    mole_H2O = (1 - wt_frac_PG) / Constant.M_H2O
    return mole_PG / (mole_PG + mole_H2O)


def get_mixture_density(mole_frac_PG, T):
    '''
    Jouyban–Acree model
    input:
        mole_frac_PG: mole fraction of PG 
        T: [K]
    output:
        rho_mixture: density of PG solution, [kg/m^3]
    '''
    rho_mixture = math.exp(mole_frac_PG * math.log(Constant.rho_PG) + 
                (1 - mole_frac_PG) * math.log(Constant.rho_H2O) + 
                27.820 * (mole_frac_PG * (1 - mole_frac_PG) / T) -
                30.537 * (mole_frac_PG * (1 - mole_frac_PG) * (mole_frac_PG - (1 - mole_frac_PG)) / T) + 
                30.476 * (mole_frac_PG * (1 - mole_frac_PG) * (mole_frac_PG - (1 - mole_frac_PG)) ** 2 / T))
    return rho_mixture


def update_wt(delta_T, wt_last, J_last, S_last):
    '''
    input:
        S_last: surface area of last time T, [m^2]
        J_last: evaporation rate of last time T, [kg/(m^2·s)]
        delta_T: time interval [s]
    '''
    evaporated_wt = J_last * S_last * delta_T
    wt_next = wt_last - evaporated_wt
    return wt_next


def updata_volume(delta_T, v_last, J_last, S_last, rho):
    '''
    input:
        S_last: surface area of last time T, [m^2]
        J_last: evaporation rate of last time T, [kg/(m^2·s)]
        delta_T: time interval [s]
    '''
    evaporated_v = J_last * S_last * delta_T / rho * 1e15
    v_next = v_last - evaporated_v
    return v_next


def get_wt_frac(wt1, wt2):
    frac1 = wt1 / (wt1 + wt2)
    return frac1


def get_contact_angle(h, bottom_radius):
    '''
    input:
        h: height of droplet
        bottom_radius: contact radius
    '''
    theta = math.atan2(h, bottom_radius)
    return math.degrees(2 * theta)


def get_surface_area(bottom_radius, h):
    '''
    input:
        R: radius of sphere, not contact radius, [um]
        h: height of droplet, [um]
        bottom_radius: contact radius
    output:
        s: surface area, [m^2]
    '''
    s = math.pi * (bottom_radius ** 2 + h ** 2) * 1e-12
    # s = 2 * math.pi * R * h * 1e-12
    return s


def solve_v2h(v, bottom_radius):
    '''
    input:
        v: volume, [pL]
        bottom_radius: contact radius
    output:
        h: height of droplet
    '''
    # print(v, bottom_radius)
    r = bottom_radius
    h = sympy.symbols('h')
    roots = sympy.solve((math.pi * h / 6) * (3 * r ** 2 + h ** 2) * 10** (-3) - v, h)
    # print(roots)
    return [root for root in roots if isinstance(root, sympy.core.numbers.Float) and root > 0][0]


def solve_R(bottom_radius, h):
    '''
    input:
        h: height of droplet
        bottom_radius: contact radius
    '''
    r = bottom_radius
    R = sympy.symbols('R')
    
    roots = sympy.solve(R ** 2 - (R - h) ** 2 - r ** 2, R)
    print(roots)
    return [root for root in roots if root > 40 ][0]


def hu_and_larson(input_path, output_path, PG_wt_frac, T, RH, delta_T):
    data = pd.read_csv(input_path)
    output_dict = {}
    
    # calculate T=0
    hu_and_larson1 = HuLarsonEq(theta=data['theta[degree]'][0], bottom_radius=data['bottom_radius[um]'][0], 
                                Mj=data['Mj[pL]'][0], T_C=T, RH=RH, wt_frac_PG=PG_wt_frac)

    recorder = hu_and_larson1.calculate()
    for item in recorder:
        output_dict[item] = []
        output_dict[item].append(recorder[item])

    recorder = hu_and_larson1.get_weight()
    for item in recorder:
        output_dict[item] = []
        output_dict[item].append(recorder[item])
    
    # calculate T=1~
    for i in range(1, len(data)):
        if output_dict['wt_frac_PG'][i-1]==0:
            wt_H2O_new = update_wt(delta_T=delta_T, wt_last=output_dict['wt_H2O'][i-1], 
                                    J_last=output_dict['J_H2O'][i-1], S_last=data['area[m^2]'][i-1])
            output_dict['wt_H2O'].append(wt_H2O_new)
            output_dict['wt_frac_PG'].append(0.0)     
        elif output_dict['wt_frac_PG'][i-1]==1:
            wt_PG_new = update_wt(delta_T=delta_T, wt_last=output_dict['wt_PG'][i-1], 
                                    J_last=output_dict['J_PG'][i-1], S_last=data['area[m^2]'][i-1])
            output_dict['wt_PG'].append(wt_PG_new)
            output_dict['wt_frac_PG'].append(1.0)
        else:
            wt_PG_new = update_wt(delta_T=delta_T, wt_last=output_dict['wt_PG'][i-1], 
                                    J_last=output_dict['J_PG'][i-1], S_last=data['area[m^2]'][i-1])
            output_dict['wt_PG'].append(wt_PG_new)

            wt_H2O_new = update_wt(delta_T=delta_T, wt_last=output_dict['wt_H2O'][i-1], 
                                    J_last=output_dict['J_H2O'][i-1], S_last=data['area[m^2]'][i-1])
            output_dict['wt_H2O'].append(wt_H2O_new)

            wt_PG_frac_new = get_wt_frac(wt_PG_new, wt_H2O_new)
            output_dict['wt_frac_PG'].append(wt_PG_frac_new)

        hu_and_larson2 = HuLarsonEq(theta=data['theta[degree]'][i], bottom_radius=data['bottom_radius[um]'][i], 
                                    Mj=data['Mj[pL]'][i], T_C=T, RH=RH, wt_frac_PG=output_dict['wt_frac_PG'][i])
        recorder = hu_and_larson2.calculate()
        for item in recorder:
            output_dict[item].append(recorder[item])
    
    # save to csv
    result = pd.DataFrame(output_dict)
    result.to_csv(output_path, index=False)


def hu_and_larson_only_theory(input_path, output_path, PG_wt_frac, T, RH, delta_T):
    data = pd.read_csv(input_path)
    output_dict = {}
    
    # calculate T=0
    hu_and_larson = HuLarsonEq(theta=data['contact angle'][0], bottom_radius=data['contact radius'][0], 
                                Mj=data['volume(pL)'][0], T_C=T, RH=RH, wt_frac_PG=PG_wt_frac)

    recorder = hu_and_larson.calculate()
    for item in recorder:
        output_dict[item] = []
        output_dict[item].append(recorder[item])

    recorder = hu_and_larson.get_weight()
    for item in recorder:
        output_dict[item] = []
        output_dict[item].append(recorder[item])
    
    output_dict['volume(pL)'] = []
    output_dict['volume(pL)'].append(data['volume(pL)'][0])
    output_dict['area(m^2)'] = []
    output_dict['area(m^2)'].append(data['area(m^2)'][0])
    output_dict['height'] = []
    output_dict['height'].append(data['height'][0])
    output_dict['contact angle'] = []
    output_dict['contact angle'].append(data['contact angle'][0])

    # calculate T=1~
    # for i in range(1, len(data)):
    for i in range(1, len(data)*5):
        if output_dict['wt_frac_PG'][i-1]==0:
            wt_H2O_new = update_wt(delta_T=delta_T, wt_last=output_dict['wt_H2O'][i-1], 
                                    J_last=output_dict['J_H2O'][i-1], S_last=data['area(m^2)'][i-1])
            output_dict['wt_H2O'].append(wt_H2O_new)
            output_dict['wt_frac_PG'].append(0.0)
            
            v_H2O_new = updata_volume(delta_T=delta_T, v_last=output_dict['volume(pL)'][i-1],
                                    J_last=output_dict['J_H2O'][i-1], S_last=output_dict['area(m^2)'][i-1], rho=Constant.rho_H2O)
            output_dict['volume(pL)'].append(v_H2O_new)

            h_new = solve_v2h(v_H2O_new, data['contact radius'][0])
            output_dict['height'].append(h_new)

            theta_new = get_contact_angle(h_new, data['contact radius'][0])
            output_dict['contact angle'].append(theta_new)
            
            S_new = get_surface_area(data['contact radius'][0], h_new)
            output_dict['area(m^2)'].append(S_new)
            
        hu_and_larson2 = HuLarsonEq(theta=output_dict['contact angle'][i], bottom_radius=data['contact radius'][0], 
                                    Mj=output_dict['volume(pL)'][i], T_C=T, RH=RH, wt_frac_PG=output_dict['wt_frac_PG'][i])
        recorder = hu_and_larson2.calculate()
        for item in recorder:
            output_dict[item].append(recorder[item])
        print('*' * 40)
        print(i, output_dict)
    # save to csv
        result = pd.DataFrame(output_dict)
        result.to_csv(output_path, index=False)



if __name__ == '__main__':
    input_path = './moriyama_data.csv'
    result_path = './/moriyama_data_testresult.csv'

    hu_and_larson_only_theory(input_path=input_path, output_path=result_path, PG_wt_frac=0, T=20.7, RH=24, delta_T=0.02)