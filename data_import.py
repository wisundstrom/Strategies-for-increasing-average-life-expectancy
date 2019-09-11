"""This script takes the raw csvs that we downloaded and reads them in and combines then as a dataframe, and
exports a csv."""
import pandas as pd
from sklearn.model_selection import train_test_split


# read in select columns from various csvs
demo = pd.read_csv('./chsi/DEMOGRAPHICS.csv', usecols=['CHSI_County_Name', 'CHSI_State_Abbr', 'Population_Size', 'State_FIPS_Code', 'County_FIPS_Code'])
birth_and_death = pd.read_csv('./chsi/MEASURESOFBIRTHANDDEATH.csv', usecols=['CHSI_County_Name', 'CHSI_State_Abbr', 'Premature', 'Under_18', 'Over_40', 'Late_Care', 'LBW', 'Suicide', 'MVA'])
vulnerable = pd.read_csv('./chsi/VUNERABLEPOPSANDENVHEALTH.csv', usecols=['CHSI_County_Name', 'CHSI_State_Abbr', 'No_HS_Diploma', 'Unemployed', 'Sev_Work_Disabled', 'Major_Depression', 'Recent_Drug_Use', 'Toxic_Chem'])
prev_serv_use = pd.read_csv('./chsi/PREVENTIVESERVICESUSE.csv', usecols=['CHSI_County_Name', 'CHSI_State_Abbr', 'Pap_Smear', 'Mammogram', 'Proctoscopy', 'Pneumo_Vax', 'Flu_Vac'])
risk_factors = pd.read_csv('./chsi/RISKFACTORSANDACCESSTOCARE.csv', usecols=['CHSI_County_Name', 'CHSI_State_Abbr', 'Uninsured', 'Prim_Care_Phys_Rate', 'Dentist_Rate', 'No_Exercise', 'Few_Fruit_Veg', 'High_Blood_Pres', 'Diabetes', 'Elderly_Medicare', 'Disabled_Medicare'])
summ_health = pd.read_csv('./chsi/SUMMARYMEASURESOFHEALTH.csv', usecols=['CHSI_County_Name', 'CHSI_State_Abbr', 'ALE'])

# merge al of the csvs together on the correct columns
chsi = pd.merge(demo, birth_and_death, left_on=['CHSI_County_Name', 'CHSI_State_Abbr'], right_on=['CHSI_County_Name', 'CHSI_State_Abbr'])
chsi = pd.merge(chsi, vulnerable, left_on=['CHSI_County_Name', 'CHSI_State_Abbr'], right_on=['CHSI_County_Name', 'CHSI_State_Abbr'])
chsi = pd.merge(chsi, prev_serv_use, left_on=['CHSI_County_Name', 'CHSI_State_Abbr'], right_on=['CHSI_County_Name', 'CHSI_State_Abbr'])
chsi = pd.merge(chsi, risk_factors, left_on=['CHSI_County_Name', 'CHSI_State_Abbr'], right_on=['CHSI_County_Name', 'CHSI_State_Abbr'])
chsi = pd.merge(chsi, summ_health, left_on=['CHSI_County_Name', 'CHSI_State_Abbr'], right_on=['CHSI_County_Name', 'CHSI_State_Abbr'])

chsi.to_csv('./data/dirty_data_full.csv')

y = pd.DataFrame(chsi['ALE'])
X = chsi.drop(columns=['ALE', 'CHSI_County_Name', 'CHSI_State_Abbr'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.to_csv('./data/dirty_X_train.csv')
y_train.to_csv('./data/dirty_y_train.csv')
X_test.to_csv('./data/dirty_X_test.csv')
y_test.to_csv('./data/dirty_y_test.csv')
