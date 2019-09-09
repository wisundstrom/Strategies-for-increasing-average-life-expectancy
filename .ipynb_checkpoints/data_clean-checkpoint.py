
def data_clean(dataset,type):
    """For type enter 'x' or 'y' to identify if is predictor or target dataset"""
    #replacing null values with the mean
    
    cols = list(dataset.columns)
    for c in cols:
        dataset.loc[dataset[c].isin([-9999,-2222,-2222.2,-2,-1111.1,-1111,-1]),c]= None
        mean = dataset[c].mean()
        dataset[c].fillna(value = mean , inplace = True)
        
    if type == 'x':
        dataset = dataset.drop(columns = ['Toxic_Chem','Pap_Smear','Proctoscopy','Flu_Vac','Pneumo_Vax','Mammogram'])
        list_totals = ['No_HS_Diploma','Unemployed','Sev_Work_Disabled','Major_Depression','Recent_Drug_Use','Uninsured' ]
        for l in list_totals:
            dataset[l] = round(((dataset[l]/dataset['Population_Size'])*100),2)
    return dataset