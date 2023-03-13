import gc

import os

import qml

import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from tqdm import tqdm



import matplotlib.pyplot as plt

stats = []

myfolder = '../input/structures/'



for filename in tqdm(os.listdir(myfolder)):

    entrystats = {}

    

    # Create the compound object mol from the file which happens to be methane

    mol = qml.Compound(xyz=myfolder+filename)

    entrystats['molecule_name'] = filename[:-4]

    mol.generate_coulomb_matrix(size=29, sorting="row-norm")

    a = mol.representation

    for i, v in enumerate(a):

        entrystats['c_'+str(i)] = a[i]

    stats.append(entrystats)

molecules = pd.DataFrame(stats)
del stats

gc.collect()
feats = list(set(molecules.columns).difference(['molecule_name']))
pca = PCA(n_components=50)

pcadata = pca.fit_transform(molecules[feats].values)

np.sum(pca.explained_variance_ratio_)
gpdata = pd.DataFrame(pcadata,columns=['pca_'+str(i) for i in range(pcadata.shape[1])])
def GPx(data):

    return (1.0*np.tanh(((((((((np.where(data["pca_1"]<0, (((data["pca_2"]) + (3.0))/2.0), data["pca_4"] )) + (3.0))) + (3.0))) - (data["pca_2"]))) + (data["pca_1"]))) +

            1.0*np.tanh(((((((((-3.0) - (((((data["pca_2"]) + (data["pca_0"]))) * ((((8.0)) - (((data["pca_0"]) / 2.0)))))))) - (np.where(data["pca_0"]<0, data["pca_0"], data["pca_0"] )))) - (data["pca_0"]))) - ((8.0)))) +

            1.0*np.tanh(((((((9.0)) + (np.where(np.where(((((9.99634075164794922)) + (np.where(data["pca_1"] < -9998, (((9.99634075164794922)) - (data["pca_2"])), (((9.99634075164794922)) - (data["pca_2"])) )))/2.0) < -9998, data["pca_1"], data["pca_2"] ) < -9998, data["pca_1"], (((9.99634075164794922)) - (data["pca_2"])) )))/2.0)) + (data["pca_1"]))) +

            1.0*np.tanh((-1.0*(((((((np.where(data["pca_0"]>0, (9.0), ((data["pca_2"]) + (data["pca_0"])) )) * 2.0)) + ((((9.0)) - (np.where(data["pca_2"]>0, ((((data["pca_2"]) / 2.0)) + (data["pca_0"])), data["pca_0"] )))))/2.0))))) +

            1.0*np.tanh(((data["pca_2"]) - (np.where((-1.0*((data["pca_0"]))) < -9998, data["pca_0"], (((10.0)) + ((((10.0)) + (((((data["pca_0"]) - (((((data["pca_40"]) - ((((10.0)) + ((-1.0*((data["pca_0"])))))))) * 2.0)))) * ((((((data["pca_2"]) / 2.0)) + (data["pca_0"]))/2.0))))))) )))) +

            1.0*np.tanh(np.where(data["pca_7"] < -9998, data["pca_1"], ((((data["pca_1"]) + ((((((np.where(data["pca_1"]<0, data["pca_7"], data["pca_1"] )) + (data["pca_11"]))/2.0)) + (np.where(data["pca_0"]<0, data["pca_4"], data["pca_1"] )))))) + (data["pca_4"])) )) +

            1.0*np.tanh(((((((data["pca_33"]) + (data["pca_1"]))) + (((data["pca_40"]) + (((((((data["pca_7"]) + (np.where(((((((data["pca_4"]) + (data["pca_34"]))/2.0)) + (data["pca_4"]))/2.0) < -9998, data["pca_1"], data["pca_40"] )))/2.0)) + (data["pca_34"]))/2.0)))))) + (data["pca_4"]))) +

            1.0*np.tanh(np.where(data["pca_0"]<0, np.where(data["pca_3"]>0, np.where(data["pca_3"]>0, 3.141593, data["pca_9"] ), np.where(data["pca_0"]>0, 3.141593, np.where(data["pca_3"]>0, 3.0, data["pca_4"] ) ) ), (((data["pca_1"]) + ((((data["pca_1"]) + ((((data["pca_3"]) + (data["pca_1"]))/2.0)))/2.0)))/2.0) )) +

            1.0*np.tanh(np.where(((((((data["pca_48"]) + ((8.0)))/2.0)) + (data["pca_0"]))/2.0)<0, data["pca_4"], ((((((((data["pca_40"]) + (((((8.0)) + (((data["pca_4"]) + (data["pca_0"]))))/2.0)))/2.0)) + (data["pca_0"]))/2.0)) - ((((((8.0)) * 2.0)) - (data["pca_48"])))) )) +

            1.0*np.tanh(np.where(data["pca_2"] < -9998, ((data["pca_1"]) / 2.0), (((data["pca_1"]) + (((((data["pca_4"]) + ((((((((data["pca_1"]) / 2.0)) - (data["pca_3"]))) + (np.where(data["pca_4"] < -9998, ((data["pca_1"]) / 2.0), data["pca_11"] )))/2.0)))) - (data["pca_2"]))))/2.0) )) +

            1.0*np.tanh((((data["pca_33"]) + (np.where(data["pca_0"]>0, data["pca_1"], (((4.0)) - (np.where(np.where(data["pca_1"]<0, data["pca_1"], ((((((7.53431272506713867)) * 2.0)) + (np.where(((1.570796) + (data["pca_1"]))<0, data["pca_33"], (11.73268222808837891) )))/2.0) )<0, data["pca_0"], (((data["pca_19"]) + (data["pca_12"]))/2.0) ))) )))/2.0)) +

            1.0*np.tanh((((((data["pca_33"]) + ((((data["pca_4"]) + (((data["pca_33"]) + ((((np.where(data["pca_7"]<0, data["pca_7"], np.where(data["pca_4"]<0, data["pca_7"], (-1.0*((data["pca_4"]))) ) )) + (np.where(data["pca_1"] < -9998, data["pca_4"], data["pca_7"] )))/2.0)))))/2.0)))) + (np.where(data["pca_0"]>0, data["pca_1"], data["pca_34"] )))/2.0)) +

            1.0*np.tanh(np.where(np.where(((data["pca_1"]) + ((10.90713787078857422)))>0, (10.90713787078857422), ((data["pca_1"]) - ((10.90713787078857422))) ) < -9998, (10.90713787078857422), ((((((10.90713787078857422)) + (data["pca_1"]))) + (np.where(((data["pca_1"]) + ((10.90713787078857422)))>0, 3.141593, (((((data["pca_7"]) + ((10.90713787078857422)))/2.0)) - (data["pca_6"])) )))/2.0) )) +

            1.0*np.tanh(np.where(data["pca_3"] < -9998, data["pca_1"], np.where(((np.where(data["pca_1"]>0, ((((data["pca_1"]) + (data["pca_0"]))) / 2.0), data["pca_1"] )) + (data["pca_3"]))>0, data["pca_0"], np.where(((data["pca_0"]) + (data["pca_3"])) < -9998, data["pca_1"], ((data["pca_1"]) + (data["pca_4"])) ) ) )) +

            1.0*np.tanh((((((((((((((((data["pca_1"]) + ((((3.0) + (0.318310))/2.0)))/2.0)) + (data["pca_0"]))/2.0)) + (data["pca_0"]))/2.0)) + ((((-1.0*((data["pca_1"])))) * (((((-1.0*((data["pca_0"])))) + (((3.0) * 2.0)))/2.0)))))/2.0)) + ((-1.0*((data["pca_39"])))))/2.0)) +

            1.0*np.tanh(np.where(data["pca_4"]<0, (((np.where(np.where(data["pca_4"] < -9998, ((data["pca_1"]) * 2.0), data["pca_1"] )>0, data["pca_40"], data["pca_7"] )) + ((((data["pca_9"]) + (np.where(data["pca_7"]>0, data["pca_40"], (((((data["pca_4"]) + (data["pca_9"]))/2.0)) * 2.0) )))/2.0)))/2.0), data["pca_1"] )) +

            1.0*np.tanh(((np.where(data["pca_0"]<0, (((-1.0*((data["pca_43"])))) / 2.0), np.where(data["pca_0"]<0, np.where(data["pca_0"]<0, ((data["pca_6"]) - (data["pca_40"])), data["pca_0"] ), ((data["pca_46"]) - ((((((data["pca_6"]) - (data["pca_40"]))) + (data["pca_38"]))/2.0))) ) )) / 2.0)) +

            1.0*np.tanh(((((np.where(data["pca_1"]<0, np.where(data["pca_6"]<0, data["pca_35"], np.where(data["pca_1"]<0, ((((data["pca_1"]) + (((data["pca_1"]) + (data["pca_3"]))))) + (data["pca_3"])), data["pca_2"] ) ), ((data["pca_40"]) - ((((data["pca_2"]) + (data["pca_3"]))/2.0))) )) / 2.0)) / 2.0)) +

            1.0*np.tanh(((((((((((((((((data["pca_46"]) - (data["pca_16"]))) - (data["pca_43"]))) - (data["pca_33"]))) + (((((((data["pca_14"]) / 2.0)) * 2.0)) - (data["pca_43"]))))/2.0)) + (np.where(data["pca_8"]<0, data["pca_11"], ((data["pca_46"]) - (data["pca_37"])) )))/2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh((((0.318310) + (np.where(data["pca_0"]<0, 0.0, np.where(data["pca_1"]<0, (((data["pca_7"]) + (data["pca_2"]))/2.0), (((data["pca_40"]) + (np.where(data["pca_1"] < -9998, 0.0, np.where(0.318310 < -9998, 0.0, np.where(data["pca_0"]<0, data["pca_40"], data["pca_35"] ) ) )))/2.0) ) )))/2.0)) +

            1.0*np.tanh(((0.318310) * ((((((((((((((((((data["pca_40"]) / 2.0)) * (data["pca_0"]))) + (data["pca_0"]))/2.0)) / 2.0)) + (np.where(data["pca_0"]>0, data["pca_9"], data["pca_47"] )))/2.0)) + ((((data["pca_47"]) + ((((data["pca_40"]) + (data["pca_48"]))/2.0)))/2.0)))/2.0)) / 2.0)))) +

            1.0*np.tanh(((np.where(-2.0 < -9998, np.where(data["pca_40"] < -9998, np.where(0.0<0, np.where(0.318310<0, (((0.0)) / 2.0), data["pca_1"] ), data["pca_46"] ), (((0.0)) / 2.0) ), ((np.where(data["pca_1"]<0, ((data["pca_42"]) / 2.0), data["pca_46"] )) / 2.0) )) / 2.0)) +

            1.0*np.tanh((((((((((((((data["pca_48"]) + (data["pca_40"]))/2.0)) - (((((data["pca_43"]) + (data["pca_38"]))) + (np.where(((-3.0) / 2.0) < -9998, data["pca_48"], data["pca_30"] )))))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(((((((-1.0*((np.where(data["pca_3"]<0, np.where(data["pca_4"]>0, 1.570796, np.where(((data["pca_4"]) / 2.0)>0, ((data["pca_4"]) / 2.0), -1.0 ) ), ((data["pca_4"]) + (data["pca_37"])) ))))) + ((((((data["pca_4"]) + (1.570796))/2.0)) / 2.0)))/2.0)) / 2.0)) +

            1.0*np.tanh(((np.where(data["pca_1"]>0, np.where(-2.0 < -9998, (((((((((0.0) / 2.0)) / 2.0)) + (0.0))/2.0)) + (data["pca_42"])), (-1.0*(((((((data["pca_41"]) + (data["pca_42"]))/2.0)) / 2.0)))) ), 0.0 )) / 2.0)) +

            1.0*np.tanh(np.where(data["pca_1"] < -9998, np.where(data["pca_43"]<0, ((((0.0) * (((data["pca_43"]) * (data["pca_48"]))))) * (data["pca_1"])), data["pca_48"] ), np.where(data["pca_1"]>0, (((0.636620) + (np.where(data["pca_43"]<0, ((0.0) * 2.0), data["pca_48"] )))/2.0), 0.0 ) )) +

            1.0*np.tanh((((((((-1.0*((np.where(data["pca_43"]<0, np.where(((data["pca_26"]) / 2.0) < -9998, ((np.where(data["pca_26"]<0, data["pca_34"], data["pca_43"] )) / 2.0), ((((((data["pca_34"]) / 2.0)) / 2.0)) / 2.0) ), np.where(data["pca_26"]<0, data["pca_34"], data["pca_43"] ) ))))) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh((((-1.0*((0.318310)))) * (((np.where(data["pca_0"]<0, ((((data["pca_35"]) / 2.0)) / 2.0), (((((data["pca_33"]) - (((((data["pca_0"]) / 2.0)) / 2.0)))) + (((0.318310) + (((0.318310) / 2.0)))))/2.0) )) / 2.0)))) +

            1.0*np.tanh(((((np.where((2.0) < -9998, ((np.where(((0.318310) * (0.0))<0, (((1.0)) / 2.0), (((-1.0*(((10.0))))) / 2.0) )) / 2.0), ((0.318310) * (np.where((-1.0*((data["pca_26"])))<0, data["pca_40"], 0.0 ))) )) / 2.0)) / 2.0)) +

            1.0*np.tanh((-1.0*((np.where(data["pca_7"]>0, 0.0, np.where(np.where((-1.0*((np.where(data["pca_10"]>0, np.where(0.0 < -9998, (0.0), 0.0 ), (-1.0*((data["pca_44"]))) ))))>0, (((0.15917305648326874)) * 2.0), data["pca_32"] )>0, 0.318310, (0.0) ) ))))) +

            1.0*np.tanh(((np.where(((0.318310) * (data["pca_9"])) < -9998, data["pca_44"], (((((np.where(((data["pca_5"]) / 2.0)<0, ((0.318310) * (data["pca_9"])), ((((data["pca_43"]) / 2.0)) * (0.636620)) )) + (((((data["pca_43"]) * (((data["pca_46"]) / 2.0)))) / 2.0)))/2.0)) / 2.0) )) / 2.0)) +

            0.999414*np.tanh((((((((np.where(data["pca_0"]<0, (((data["pca_33"]) + (data["pca_21"]))/2.0), np.where(0.318310 < -9998, 3.141593, (((data["pca_4"]) + (np.where(np.where(data["pca_13"]>0, data["pca_33"], data["pca_28"] )>0, (((0.0) + (data["pca_33"]))/2.0), data["pca_33"] )))/2.0) ) )) + (0.318310))/2.0)) / 2.0)) * (0.318310))) +

            1.0*np.tanh((-1.0*((((np.where(data["pca_0"]>0, (((data["pca_0"]) + ((((data["pca_2"]) + (((((1.570796) + (((1.570796) + (data["pca_2"]))))) * ((((((data["pca_0"]) + (((((-3.0) * 2.0)) * 2.0)))/2.0)) * 2.0)))))/2.0)))/2.0), 0.0 )) / 2.0))))) +

            1.0*np.tanh(np.where(np.where(data["pca_5"]>0, np.where(0.318310>0, data["pca_2"], data["pca_4"] ), data["pca_5"] )>0, ((-1.0) + (0.318310)), np.where(data["pca_4"]>0, (((((((0.15468123555183411)) / 2.0)) / 2.0)) * (data["pca_2"])), 0.318310 ) )) +

            1.0*np.tanh(((((np.where(np.where(data["pca_8"]>0, data["pca_28"], 3.0 )<0, np.where(3.0>0, (((data["pca_28"]) + ((((data["pca_8"]) + (((data["pca_46"]) / 2.0)))/2.0)))/2.0), (((data["pca_28"]) + ((((((0.0) / 2.0)) + (data["pca_8"]))/2.0)))/2.0) ), 0.636620 )) / 2.0)) / 2.0)) +

            0.901543*np.tanh(((np.where(((0.0) / 2.0) < -9998, 0.0, ((((np.where(((data["pca_10"]) / 2.0)>0, 0.636620, ((np.where(((((data["pca_15"]) / 2.0)) / 2.0) < -9998, data["pca_15"], 0.318310 )) * (((data["pca_15"]) - (((0.0) + (data["pca_40"])))))) )) / 2.0)) / 2.0) )) / 2.0)) +

            0.999414*np.tanh((((-1.0*((np.where(data["pca_13"]>0, 0.318310, ((((((-1.0*((0.318310)))) + ((((((data["pca_42"]) + ((-1.0*((np.where(data["pca_13"]>0, 0.318310, (((0.0) + (0.318310))/2.0) ))))))) + (((((0.318310) / 2.0)) * (0.0))))/2.0)))/2.0)) / 2.0) ))))) / 2.0)) +

            0.962883*np.tanh(0.0) +

            1.0*np.tanh(((((((((((((data["pca_47"]) + (data["pca_46"]))/2.0)) + (((((data["pca_44"]) * ((((data["pca_47"]) + (np.where(((((((((((data["pca_46"]) + (((0.0) / 2.0)))/2.0)) / 2.0)) / 2.0)) + (data["pca_46"]))/2.0) < -9998, ((data["pca_46"]) / 2.0), data["pca_46"] )))/2.0)))) / 2.0)))/2.0)) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(np.where(data["pca_19"]<0, ((0.318310) / 2.0), ((((np.where((-1.0*((data["pca_6"])))<0, 0.318310, -1.0 )) / 2.0)) / 2.0) )) +

            0.878101*np.tanh((((0.07979395240545273)) * ((((((((((((data["pca_17"]) + (data["pca_36"]))/2.0)) - (data["pca_20"]))) + (data["pca_30"]))/2.0)) + (np.where(0.0 < -9998, ((data["pca_7"]) * (1.570796)), (((0.07979395240545273)) * ((((data["pca_17"]) + (data["pca_36"]))/2.0))) )))/2.0)))) +

            1.0*np.tanh(((((np.where(data["pca_24"] < -9998, data["pca_24"], (((((((((data["pca_25"]) / 2.0)) / 2.0)) + ((((((data["pca_24"]) + (((((((((data["pca_4"]) / 2.0)) + (((data["pca_24"]) / 2.0)))/2.0)) + (data["pca_4"]))/2.0)))/2.0)) / 2.0)))/2.0)) / 2.0) )) / 2.0)) / 2.0)) +

            1.0*np.tanh(((((((((data["pca_39"]) * ((-1.0*((np.where(data["pca_7"] < -9998, ((data["pca_7"]) / 2.0), ((((((((((data["pca_37"]) + ((((data["pca_7"]) + (data["pca_39"]))/2.0)))/2.0)) / 2.0)) + (0.636620))/2.0)) / 2.0) ))))))) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(np.where(data["pca_22"]>0, ((np.where(data["pca_29"]>0, (-1.0*((((0.318310) / 2.0)))), np.where(((0.0) / 2.0)>0, 0.0, ((((-1.0*((0.0)))) + (0.0))/2.0) ) )) / 2.0), (((0.0) + ((((0.318310) + (0.0))/2.0)))/2.0) )) +

            0.999805*np.tanh(((np.where((((np.where(data["pca_2"]<0, data["pca_10"], (((((((-1.0*((data["pca_11"])))) + (data["pca_16"]))/2.0)) + ((-1.0*((data["pca_24"])))))/2.0) )) + (data["pca_46"]))/2.0)<0, (-1.0*((np.where(data["pca_16"]>0, 0.636620, 0.318310 )))), np.where(data["pca_16"]>0, 0.636620, 0.318310 ) )) / 2.0)) +

            0.932213*np.tanh((-1.0*((((((0.318310) * (0.318310))) * (0.318310)))))) +

            1.0*np.tanh(np.where((((1.570796) + (((((((((((((1.570796) + (data["pca_4"]))/2.0)) + (data["pca_0"]))/2.0)) + (data["pca_3"]))/2.0)) + (data["pca_4"]))/2.0)))/2.0)>0, 0.0, (((1.570796) + (((((((1.570796) + (data["pca_4"]))/2.0)) + (((1.570796) * 2.0)))/2.0)))/2.0) )) +

            1.0*np.tanh(((((np.where(((3.141593) - (data["pca_0"]))<0, ((np.where(3.141593<0, ((data["pca_5"]) / 2.0), (-1.0*((((((data["pca_42"]) - (data["pca_5"]))) / 2.0)))) )) / 2.0), np.where(data["pca_0"]<0, -1.0, ((3.141593) * (((3.141593) - (data["pca_1"])))) ) )) / 2.0)) / 2.0)) +

            1.0*np.tanh(((((((((np.where(data["pca_34"]<0, np.where(data["pca_4"]>0, data["pca_34"], data["pca_12"] ), np.where(data["pca_0"]<0, np.where(data["pca_4"]>0, (-1.0*((data["pca_12"]))), (5.0) ), data["pca_4"] ) )) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            0.848017*np.tanh(((0.0) - ((((((((((((((((data["pca_3"]) + (((data["pca_3"]) * (((((((data["pca_33"]) + (((data["pca_19"]) - (data["pca_12"]))))/2.0)) + ((((data["pca_33"]) + (((data["pca_19"]) - (data["pca_2"]))))/2.0)))/2.0)))))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)))) +

            0.940223*np.tanh(((0.318310) * (((((((((((((data["pca_24"]) - ((((data["pca_12"]) + (data["pca_10"]))/2.0)))) + (data["pca_17"]))/2.0)) + ((-1.0*(((((((np.where(data["pca_5"]<0, data["pca_8"], data["pca_12"] )) + (((data["pca_4"]) + ((((data["pca_24"]) + (data["pca_5"]))/2.0)))))/2.0)) / 2.0))))))/2.0)) / 2.0)) / 2.0)))) +

            1.0*np.tanh((-1.0*((((((((((np.where(((((((((data["pca_44"]) + (((data["pca_44"]) / 2.0)))) / 2.0)) / 2.0)) / 2.0) < -9998, ((data["pca_28"]) + (((0.0) * 2.0))), ((((data["pca_44"]) + (((data["pca_29"]) / 2.0)))) / 2.0) )) / 2.0)) / 2.0)) / 2.0)) / 2.0))))) +

            0.890018*np.tanh((-1.0*((((np.where(data["pca_6"]<0, np.where(0.318310<0, (0.0), np.where(((data["pca_33"]) / 2.0)<0, np.where(data["pca_28"]<0, 0.636620, np.where((0.0) < -9998, ((0.318310) / 2.0), ((0.0) * 2.0) ) ), np.where(data["pca_43"]<0, 0.636620, (0.0) ) ) ), (0.0) )) / 2.0))))) +

            0.971283*np.tanh(((np.where(np.where(((((0.0) * 2.0)) / 2.0)<0, np.where(data["pca_3"]>0, data["pca_35"], np.where((0.0)<0, 0.0, np.where(0.0>0, data["pca_35"], data["pca_48"] ) ) ), np.where(data["pca_3"]>0, data["pca_35"], data["pca_48"] ) )<0, 0.318310, 0.0 )) / 2.0)) +

            0.841375*np.tanh(np.where((((data["pca_5"]) + (((((np.where(data["pca_5"] < -9998, data["pca_1"], -2.0 )) * 2.0)) - (data["pca_1"]))))/2.0)<0, np.where((((data["pca_4"]) + (data["pca_1"]))/2.0)<0, 0.636620, np.where((((data["pca_1"]) + (data["pca_2"]))/2.0)<0, data["pca_1"], (-1.0*((0.318310))) ) ), (-1.0*((0.318310))) )) +

            0.999609*np.tanh(np.where((((data["pca_5"]) + ((((data["pca_4"]) + (data["pca_11"]))/2.0)))/2.0)>0, ((0.318310) / 2.0), np.where((((data["pca_2"]) + (data["pca_4"]))/2.0)>0, 0.318310, ((((0.318310) * ((((((data["pca_11"]) + (data["pca_4"]))/2.0)) / 2.0)))) / 2.0) ) )) +

            1.0*np.tanh(0.0) +

            1.0*np.tanh((((((0.06277085840702057)) / 2.0)) * ((((((-1.0*((((np.where(data["pca_42"]<0, data["pca_7"], ((np.where(data["pca_34"]<0, data["pca_7"], (-1.0*(((((-1.0*((0.0)))) / 2.0)))) )) / 2.0) )) / 2.0))))) / 2.0)) * ((-1.0*((data["pca_42"])))))))) +

            0.879078*np.tanh(((((((((((((((0.318310) / 2.0)) + (0.0))/2.0)) / 2.0)) + (((np.where((((((np.where(data["pca_1"]<0, 0.0, ((data["pca_33"]) - (((data["pca_6"]) + (data["pca_45"])))) )) / 2.0)) + (data["pca_45"]))/2.0)>0, -1.0, ((data["pca_1"]) / 2.0) )) / 2.0)))/2.0)) / 2.0)) / 2.0)) +

            0.998047*np.tanh(0.0) +

            1.0*np.tanh(0.0) +

            0.999414*np.tanh(((np.where(((0.0) / 2.0) < -9998, ((((((((data["pca_49"]) / 2.0)) / 2.0)) / 2.0)) / 2.0), 0.318310 )) * (((np.where((0.0)>0, np.where(0.0>0, ((((data["pca_49"]) / 2.0)) * 2.0), 0.318310 ), 0.318310 )) * (((((data["pca_49"]) / 2.0)) / 2.0)))))) +

            1.0*np.tanh((((-1.0*((data["pca_1"])))) * (((((((((np.where(0.318310 < -9998, ((((0.318310) * ((-1.0*((data["pca_1"])))))) * (data["pca_6"])), (-1.0*((((((data["pca_6"]) * (((0.318310) * (0.318310))))) / 2.0)))) )) / 2.0)) / 2.0)) * (0.318310))) * (0.318310))))) +

            1.0*np.tanh((((((0.0) + (((0.318310) * ((-1.0*((np.where((((data["pca_2"]) + (((data["pca_14"]) / 2.0)))/2.0)>0, 0.0, (((((((((0.318310) + (data["pca_6"]))/2.0)) + (data["pca_1"]))/2.0)) + ((((data["pca_9"]) + (data["pca_7"]))/2.0)))/2.0) ))))))))/2.0)) / 2.0)) +

            0.953702*np.tanh(((((((np.where(data["pca_2"]>0, 0.0, np.where(data["pca_3"]>0, (((((data["pca_2"]) + (data["pca_5"]))/2.0)) / 2.0), (((data["pca_27"]) + (((data["pca_13"]) - ((((data["pca_23"]) + ((((data["pca_2"]) + (data["pca_5"]))/2.0)))/2.0)))))/2.0) ) )) / 2.0)) / 2.0)) / 2.0)) +

            0.916976*np.tanh(((((np.where((((data["pca_2"]) + (data["pca_36"]))/2.0)<0, 0.318310, (-1.0*((((((((((data["pca_13"]) + ((((data["pca_8"]) + (((((((data["pca_2"]) + ((((((data["pca_7"]) * (data["pca_8"]))) + (data["pca_13"]))/2.0)))/2.0)) + (data["pca_7"]))/2.0)))/2.0)))/2.0)) + (data["pca_39"]))/2.0)) / 2.0)))) )) / 2.0)) / 2.0)) +

            1.0*np.tanh(((np.where(data["pca_0"]>0, ((0.318310) / 2.0), np.where(data["pca_33"]>0, 0.0, np.where(data["pca_42"]>0, (-1.0*(((-1.0*((0.318310)))))), (-1.0*((0.318310))) ) ) )) / 2.0)) +

            1.0*np.tanh((-1.0*((np.where(0.0 < -9998, 0.0, np.where(0.0<0, (-1.0*((data["pca_3"]))), np.where(data["pca_11"]<0, (0.0), np.where((-1.0*((data["pca_3"])))<0, ((0.0) * 2.0), np.where((-1.0*((data["pca_30"])))<0, 0.0, 0.318310 ) ) ) ) ))))) +

            0.952530*np.tanh((((((((-1.0*(((((((((((((data["pca_19"]) + (((data["pca_5"]) * (data["pca_48"]))))/2.0)) + (((data["pca_46"]) * (data["pca_48"]))))/2.0)) + ((((((data["pca_33"]) + (((((((data["pca_48"]) + (data["pca_4"]))/2.0)) + (data["pca_1"]))/2.0)))/2.0)) * (data["pca_48"]))))/2.0)) / 2.0))))) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh((((-1.0*(((((0.05990625917911530)) / 2.0))))) * ((((((data["pca_45"]) * (data["pca_37"]))) + ((((((((data["pca_2"]) / 2.0)) * (data["pca_40"]))) + (((np.where(data["pca_1"]>0, np.where(data["pca_2"]>0, data["pca_1"], data["pca_6"] ), data["pca_28"] )) * (data["pca_37"]))))/2.0)))/2.0)))) +

            1.0*np.tanh(((((((((0.318310) / 2.0)) / 2.0)) * (data["pca_41"]))) * (((np.where(((((((0.318310) / 2.0)) / 2.0)) / 2.0)<0, 0.636620, np.where(data["pca_13"]<0, ((0.318310) * (((0.318310) / 2.0))), 0.636620 ) )) / 2.0)))) +

            0.976167*np.tanh(((0.0) / 2.0)) +

            0.999805*np.tanh(((((((((((np.where(data["pca_3"] < -9998, data["pca_6"], np.where(data["pca_2"]<0, np.where(data["pca_1"]<0, data["pca_28"], data["pca_3"] ), ((data["pca_12"]) * (((((((np.where(data["pca_28"]<0, data["pca_6"], data["pca_3"] )) / 2.0)) / 2.0)) / 2.0))) ) )) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            0.816761*np.tanh(0.0) +

            1.0*np.tanh(np.where(data["pca_35"]>0, 0.0, ((np.where(((data["pca_3"]) - (data["pca_2"]))>0, np.where(data["pca_2"]>0, ((2.0) + (((2.0) + (data["pca_0"])))), ((0.0) * 2.0) ), ((((np.where(data["pca_0"]<0, 2.0, ((data["pca_3"]) / 2.0) )) / 2.0)) / 2.0) )) / 2.0) )) +

            1.0*np.tanh(((((0.0) / 2.0)) * 2.0)) +

            1.0*np.tanh(0.0) +

            0.888259*np.tanh(((np.where(data["pca_13"]<0, (0.0), ((((np.where(((np.where((0.0)<0, 0.0, 0.318310 )) / 2.0)<0, data["pca_13"], 0.318310 )) * ((((((2.0) + (data["pca_12"]))/2.0)) / 2.0)))) / 2.0) )) / 2.0)) +

            1.0*np.tanh((((0.02393961511552334)) * ((-1.0*((np.where(data["pca_4"]>0, (0.02393961511552334), np.where(data["pca_1"]>0, np.where(data["pca_27"]>0, np.where(data["pca_32"]>0, data["pca_6"], ((data["pca_1"]) / 2.0) ), np.where(((data["pca_1"]) / 2.0)>0, data["pca_11"], data["pca_6"] ) ), (-1.0*((data["pca_1"]))) ) ))))))) +

            1.0*np.tanh(np.where(np.where(data["pca_6"]<0, data["pca_6"], data["pca_32"] )<0, np.where(data["pca_2"]<0, ((np.where(data["pca_33"]<0, (0.0), (1.0) )) / 2.0), ((np.where(data["pca_33"]>0, ((((0.318310) * (((data["pca_6"]) / 2.0)))) / 2.0), (0.0) )) / 2.0) ), (0.0) )) +

            0.999609*np.tanh((((((((((((((np.where((((data["pca_26"]) + (data["pca_0"]))/2.0)<0, data["pca_26"], ((((((((((np.where(data["pca_46"] < -9998, data["pca_46"], data["pca_26"] )) / 2.0)) + (data["pca_0"]))/2.0)) + (data["pca_0"]))/2.0)) / 2.0) )) / 2.0)) + ((-1.0*((data["pca_46"])))))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            0.999805*np.tanh(((0.318310) * (((np.where(0.0 < -9998, ((((((data["pca_47"]) / 2.0)) / 2.0)) - ((-1.0*((0.318310))))), ((((((0.318310) - (((((data["pca_40"]) - (((data["pca_47"]) * 2.0)))) / 2.0)))) / 2.0)) / 2.0) )) / 2.0)))) +

            0.917367*np.tanh((-1.0*((np.where(((((data["pca_4"]) - (data["pca_45"]))) - (data["pca_2"]))<0, 0.0, np.where(data["pca_3"]<0, ((np.where(np.where(data["pca_3"] < -9998, data["pca_2"], ((((data["pca_4"]) - (data["pca_2"]))) - (data["pca_2"])) )<0, data["pca_4"], 0.0 )) / 2.0), 0.318310 ) ))))) +

            0.902911*np.tanh((((((0.11011722683906555)) * ((((0.11011722683906555)) - (((np.where(((data["pca_1"]) - (data["pca_7"]))<0, ((((data["pca_14"]) - (data["pca_11"]))) - (data["pca_5"])), (((((data["pca_11"]) + (data["pca_5"]))) + (((((data["pca_11"]) + ((0.11011722683906555)))) + (data["pca_7"]))))/2.0) )) / 2.0)))))) / 2.0)) +

            1.0*np.tanh(np.where(0.0<0, np.where(0.0>0, 0.318310, 0.318310 ), ((((0.318310) * (np.where(np.where(data["pca_2"]>0, 0.318310, data["pca_1"] )>0, 0.318310, (((((data["pca_10"]) + (np.where(data["pca_11"]<0, (((data["pca_10"]) + (data["pca_11"]))/2.0), 0.318310 )))/2.0)) / 2.0) )))) / 2.0) )) +

            1.0*np.tanh(0.0) +

            0.976363*np.tanh(((((((((((np.where(data["pca_34"] < -9998, data["pca_4"], np.where(data["pca_4"]>0, data["pca_43"], ((((((data["pca_4"]) + (data["pca_6"]))/2.0)) + (((data["pca_34"]) + (((data["pca_14"]) + ((((((data["pca_2"]) * (data["pca_43"]))) + (data["pca_34"]))/2.0)))))))/2.0) ) )) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(np.where((((((((data["pca_43"]) + (((((((data["pca_43"]) + (((data["pca_43"]) + (((data["pca_0"]) - (data["pca_30"]))))))) + (((data["pca_10"]) * 2.0)))) * (((data["pca_0"]) * (((data["pca_0"]) - (data["pca_1"]))))))))/2.0)) / 2.0)) * (data["pca_43"])) < -9998, data["pca_1"], (0.0) )) +

            1.0*np.tanh(((((((((((((np.where(np.where(((-1.0) / 2.0) < -9998, data["pca_33"], ((-1.0) / 2.0) )<0, ((np.where(data["pca_33"]<0, data["pca_33"], -1.0 )) / 2.0), ((((0.0) / 2.0)) / 2.0) )) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            0.848017*np.tanh(((((np.where(((0.0) / 2.0) < -9998, ((0.0) / 2.0), np.where((((data["pca_1"]) + (-1.0))/2.0)<0, 0.636620, -1.0 ) )) / 2.0)) / 2.0)) +

            1.0*np.tanh(0.0) +

            0.845282*np.tanh(np.where(0.318310<0, 0.318310, np.where(data["pca_3"]<0, 0.0, np.where(data["pca_37"]>0, np.where(data["pca_3"]>0, ((0.318310) / 2.0), (-1.0*((0.318310))) ), np.where(((0.0) * (0.0))<0, (-1.0*((0.318310))), (-1.0*((0.318310))) ) ) ) )) +

            1.0*np.tanh(0.0) +

            1.0*np.tanh(0.0) +

            1.0*np.tanh(0.0) +

            0.999609*np.tanh(((((np.where(data["pca_4"]<0, ((((0.0) / 2.0)) / 2.0), 0.0 )) / 2.0)) * (np.where(((data["pca_4"]) / 2.0)>0, 0.0, data["pca_40"] )))) +

            0.999609*np.tanh(((((0.318310) * (((((0.318310) * (np.where(data["pca_40"]<0, (-1.0*((np.where(data["pca_2"]<0, ((data["pca_43"]) / 2.0), np.where(data["pca_8"]>0, data["pca_4"], data["pca_8"] ) )))), np.where(data["pca_2"]>0, data["pca_29"], data["pca_8"] ) )))) / 2.0)))) / 2.0)) +

            1.0*np.tanh((0.0)) +

            1.0*np.tanh((((((0.02373934350907803)) * (((data["pca_28"]) / 2.0)))) * (((((((-1.0*((data["pca_6"])))) + (np.where(np.where(data["pca_6"]<0, data["pca_28"], (((-1.0*((data["pca_3"])))) / 2.0) )<0, data["pca_4"], (-1.0*((data["pca_11"]))) )))/2.0)) / 2.0)))) +

            0.999805*np.tanh((((-1.0*((((((np.where(data["pca_6"]<0, data["pca_47"], (((np.where(data["pca_7"]<0, ((data["pca_23"]) + (np.where(data["pca_16"]<0, data["pca_7"], data["pca_6"] ))), data["pca_29"] )) + (np.where(data["pca_7"]<0, data["pca_7"], data["pca_6"] )))/2.0) )) / 2.0)) / 2.0))))) * (((0.318310) / 2.0)))))



def GPy(data):

    return (1.0*np.tanh(((((data["pca_4"]) + (data["pca_0"]))) + ((((np.where(data["pca_2"]>0, data["pca_7"], (((((data["pca_1"]) + (((data["pca_4"]) + (data["pca_0"]))))/2.0)) + (((data["pca_0"]) + ((6.0))))) )) + ((12.70757579803466797)))/2.0)))) +

            1.0*np.tanh(((data["pca_0"]) + ((((np.where(data["pca_0"] < -9998, np.where(((data["pca_25"]) - (data["pca_7"])) < -9998, 3.141593, -3.0 ), (9.64634513854980469) )) + ((((((data["pca_7"]) + (((data["pca_4"]) - (((data["pca_8"]) - ((((data["pca_4"]) + (data["pca_11"]))/2.0)))))))/2.0)) * 2.0)))/2.0)))) +

            1.0*np.tanh(((((data["pca_37"]) + (np.where((((-1.0*((1.570796)))) * 2.0) < -9998, data["pca_37"], (((((((data["pca_11"]) + (data["pca_0"]))) + (((data["pca_1"]) + (data["pca_28"]))))/2.0)) - (data["pca_5"])) )))) + (((((data["pca_7"]) + (data["pca_0"]))) - (data["pca_8"]))))) +

            1.0*np.tanh((((((data["pca_0"]) - ((-1.0*(((((((np.where(data["pca_15"] < -9998, data["pca_11"], (((data["pca_0"]) + ((((-1.0*(((((-1.0*((((data["pca_11"]) - (data["pca_8"])))))) - ((-1.0*((data["pca_7"]))))))))) / 2.0)))/2.0) )) + (((data["pca_11"]) * 2.0)))/2.0)) - (data["pca_8"])))))))) + (data["pca_7"]))/2.0)) +

            1.0*np.tanh(((((data["pca_0"]) + (np.where(data["pca_2"]<0, np.where(((data["pca_0"]) + (np.where((9.0)<0, np.where(data["pca_0"]>0, data["pca_0"], 3.0 ), np.where((((4.0)) * (data["pca_0"]))<0, data["pca_7"], data["pca_0"] ) )))<0, (10.0), (4.0) ), data["pca_7"] )))) + (3.0))) +

            1.0*np.tanh(np.where(data["pca_1"] < -9998, (((((data["pca_4"]) + (data["pca_0"]))/2.0)) + ((8.0))), ((((np.where(data["pca_1"]<0, data["pca_4"], data["pca_0"] )) + ((((((8.0)) + (data["pca_0"]))) + ((8.0)))))) + (data["pca_4"])) )) +

            1.0*np.tanh(((((((((data["pca_4"]) + ((((5.0)) + (np.where(np.where((1.01141476631164551)<0, np.where(data["pca_4"] < -9998, 3.0, data["pca_1"] ), ((data["pca_1"]) / 2.0) )<0, data["pca_4"], data["pca_0"] )))))) + (((data["pca_0"]) + (data["pca_40"]))))/2.0)) + (np.where(data["pca_0"] < -9998, data["pca_1"], (5.0) )))/2.0)) +

            1.0*np.tanh(((((-1.0*((((data["pca_8"]) + (((data["pca_46"]) * 2.0))))))) + (np.where(data["pca_2"]<0, data["pca_0"], (((((np.where(((data["pca_0"]) * 2.0)>0, data["pca_1"], ((data["pca_0"]) - (np.where(data["pca_0"]>0, data["pca_1"], data["pca_5"] ))) )) + (data["pca_11"]))) + (data["pca_4"]))/2.0) )))/2.0)) +

            1.0*np.tanh((((data["pca_37"]) + ((((data["pca_28"]) + (((data["pca_0"]) + (np.where(data["pca_0"] < -9998, np.where(data["pca_2"]<0, data["pca_0"], np.where(data["pca_0"]<0, data["pca_4"], (-1.0*((data["pca_2"]))) ) ), np.where(data["pca_0"]<0, data["pca_4"], (-1.0*((data["pca_2"]))) ) )))))/2.0)))/2.0)) +

            1.0*np.tanh(np.where(data["pca_7"] < -9998, data["pca_0"], np.where(data["pca_1"]<0, data["pca_7"], ((((3.141593) - (((data["pca_8"]) + ((((data["pca_0"]) + (data["pca_1"]))/2.0)))))) + (np.where((((data["pca_0"]) + (data["pca_1"]))/2.0)<0, ((data["pca_0"]) + (data["pca_0"])), data["pca_1"] ))) ) )) +

            1.0*np.tanh(((((((12.12216091156005859)) + (np.where(data["pca_11"]>0, data["pca_0"], ((data["pca_1"]) * (((data["pca_0"]) / 2.0))) )))) + ((((((data["pca_4"]) * 2.0)) + (np.where((((12.12216091156005859)) + (data["pca_0"]))>0, (((((12.12216091156005859)) + (data["pca_1"]))) + (data["pca_4"])), data["pca_11"] )))/2.0)))/2.0)) +

            1.0*np.tanh(((np.where(((data["pca_0"]) - (data["pca_2"]))<0, data["pca_4"], np.where(data["pca_1"] < -9998, data["pca_0"], -2.0 ) )) - (np.where(data["pca_0"]<0, data["pca_1"], np.where(data["pca_1"]<0, ((data["pca_2"]) - (-2.0)), ((-2.0) + (((-2.0) + (data["pca_40"])))) ) )))) +

            1.0*np.tanh((((data["pca_38"]) + (np.where(data["pca_0"]<0, ((((((data["pca_0"]) - (data["pca_7"]))) - (data["pca_38"]))) - (data["pca_7"])), (((data["pca_42"]) + (np.where(((data["pca_0"]) - (data["pca_7"]))<0, data["pca_29"], (((-1.0*((data["pca_6"])))) - (data["pca_4"])) )))/2.0) )))/2.0)) +

            1.0*np.tanh(((((((-1.0*((data["pca_8"])))) + (((((((((data["pca_4"]) + (((((((data["pca_4"]) + (data["pca_0"]))/2.0)) + ((((data["pca_4"]) + (data["pca_29"]))/2.0)))/2.0)))/2.0)) + (((data["pca_11"]) - (data["pca_5"]))))/2.0)) + ((((data["pca_38"]) + ((((data["pca_29"]) + (data["pca_0"]))/2.0)))/2.0)))))/2.0)) + (data["pca_41"]))) +

            1.0*np.tanh(np.where(((data["pca_0"]) / 2.0)<0, (((data["pca_4"]) + (data["pca_2"]))/2.0), np.where(0.318310 < -9998, ((data["pca_0"]) / 2.0), np.where(((((data["pca_0"]) / 2.0)) + (data["pca_2"]))>0, (-1.0*((((data["pca_40"]) + (data["pca_46"]))))), data["pca_0"] ) ) )) +

            1.0*np.tanh(np.where((((3.141593) + (data["pca_4"]))/2.0)>0, (((((data["pca_33"]) + (data["pca_1"]))/2.0)) + (np.where(data["pca_2"]<0, data["pca_2"], 3.141593 ))), (((data["pca_6"]) + (np.where(data["pca_2"] < -9998, np.where(data["pca_1"]<0, data["pca_2"], data["pca_33"] ), ((data["pca_0"]) - (data["pca_2"])) )))/2.0) )) +

            1.0*np.tanh(((np.where(data["pca_9"]>0, (((((((-1.0*(((((data["pca_40"]) + (data["pca_2"]))/2.0))))) + (((data["pca_28"]) - (((((data["pca_40"]) + (data["pca_46"]))) + (data["pca_7"]))))))/2.0)) + (data["pca_37"]))/2.0), (-1.0*((data["pca_7"]))) )) / 2.0)) +

            1.0*np.tanh((((((((data["pca_29"]) / 2.0)) + ((((data["pca_18"]) + (((np.where(data["pca_17"]>0, np.where(data["pca_17"]>0, data["pca_38"], data["pca_0"] ), data["pca_0"] )) - (((data["pca_23"]) - (np.where(data["pca_17"]>0, data["pca_44"], 0.0 )))))))/2.0)))/2.0)) / 2.0)) +

            1.0*np.tanh((((-1.0*(((((data["pca_46"]) + (np.where(data["pca_42"] < -9998, data["pca_46"], (((data["pca_45"]) + ((((data["pca_6"]) + (((data["pca_9"]) - (((data["pca_42"]) * 2.0)))))/2.0)))/2.0) )))/2.0))))) / 2.0)) +

            1.0*np.tanh(((data["pca_40"]) * ((-1.0*((np.where(data["pca_1"]>0, 0.636620, (-1.0*((np.where(data["pca_40"]>0, 0.318310, (((-1.0*((np.where(data["pca_1"]>0, 0.636620, (-1.0*((np.where(data["pca_40"]>0, 0.318310, 0.0 )))) ))))) * (data["pca_40"])) )))) ))))))) +

            1.0*np.tanh(((((((((data["pca_34"]) + (np.where(data["pca_1"]<0, data["pca_8"], np.where(data["pca_1"] < -9998, 0.0, data["pca_38"] ) )))/2.0)) + (((np.where(data["pca_8"]<0, data["pca_44"], ((data["pca_41"]) - (data["pca_10"])) )) - (data["pca_47"]))))/2.0)) / 2.0)) +

            1.0*np.tanh(((np.where(data["pca_0"]<0, np.where(data["pca_1"]>0, np.where(data["pca_1"]>0, (-1.0*((np.where(data["pca_1"] < -9998, data["pca_4"], ((data["pca_5"]) - (((data["pca_4"]) / 2.0))) )))), data["pca_1"] ), data["pca_6"] ), (-1.0*((np.where(data["pca_8"]>0, data["pca_46"], ((data["pca_4"]) / 2.0) )))) )) / 2.0)) +

            1.0*np.tanh((((((((np.where(data["pca_8"]>0, (((data["pca_38"]) + (data["pca_38"]))/2.0), data["pca_7"] )) + ((((-3.0) + (((((-1.0*((data["pca_8"])))) + (((((((-1.0*((data["pca_37"])))) + ((-1.0*((data["pca_8"])))))/2.0)) * (((data["pca_1"]) / 2.0)))))/2.0)))/2.0)))/2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(((np.where(0.0 < -9998, data["pca_8"], np.where(((data["pca_8"]) + (data["pca_2"]))<0, 0.318310, (-1.0*((((np.where(data["pca_40"] < -9998, data["pca_21"], (((((data["pca_47"]) - (data["pca_39"]))) + (np.where(data["pca_8"] < -9998, 0.318310, ((data["pca_46"]) - (data["pca_39"])) )))/2.0) )) / 2.0)))) ) )) / 2.0)) +

            1.0*np.tanh(np.where((((-1.0*((data["pca_1"])))) / 2.0)<0, (((-1.0) + (np.where((((-1.0*((data["pca_0"])))) / 2.0)<0, ((data["pca_43"]) - (data["pca_40"])), (((-1.0*((data["pca_11"])))) / 2.0) )))/2.0), np.where(data["pca_43"] < -9998, np.where(data["pca_1"]>0, data["pca_0"], data["pca_11"] ), 0.0 ) )) +

            1.0*np.tanh(((((((((((((((data["pca_30"]) / 2.0)) + ((((data["pca_1"]) + (((np.where(data["pca_41"]>0, data["pca_41"], data["pca_43"] )) - (data["pca_0"]))))/2.0)))/2.0)) / 2.0)) + (np.where(data["pca_4"]>0, data["pca_43"], ((data["pca_41"]) - (np.where(data["pca_0"]>0, data["pca_22"], data["pca_1"] ))) )))/2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh((((((((((data["pca_44"]) + (((((((data["pca_20"]) + (((((data["pca_38"]) * (data["pca_9"]))) / 2.0)))/2.0)) + (np.where((-1.0*((data["pca_35"]))) < -9998, ((((data["pca_9"]) * (data["pca_9"]))) / 2.0), (-1.0*((data["pca_14"]))) )))/2.0)))/2.0)) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(np.where(data["pca_1"]>0, 0.0, ((np.where(0.0>0, data["pca_0"], np.where(data["pca_0"]<0, 0.0, ((((((data["pca_5"]) / 2.0)) + (((((((data["pca_9"]) + (data["pca_0"]))/2.0)) + (data["pca_4"]))/2.0)))) + (((0.0) / 2.0))) ) )) / 2.0) )) +

            1.0*np.tanh(np.where(((((data["pca_1"]) + (data["pca_2"]))) + (np.where(data["pca_2"]>0, (0.0), data["pca_37"] )))>0, 0.0, ((data["pca_2"]) * (np.where(data["pca_5"]<0, data["pca_0"], (((data["pca_37"]) + (((data["pca_1"]) + (np.where(data["pca_1"]>0, -2.0, data["pca_4"] )))))/2.0) ))) )) +

            1.0*np.tanh((((((((-2.0) + ((((-2.0) + ((((((data["pca_33"]) + (np.where(data["pca_33"] < -9998, (-1.0*((data["pca_7"]))), ((((((((((data["pca_1"]) + (data["pca_38"]))/2.0)) + (data["pca_16"]))/2.0)) / 2.0)) * ((((data["pca_1"]) + (data["pca_10"]))/2.0))) )))/2.0)) / 2.0)))/2.0)))/2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(((0.318310) * ((((0.318310) + (((((((((0.636620) + (data["pca_37"]))/2.0)) / 2.0)) + ((((-1.0*((((0.636620) + ((((data["pca_3"]) + (((((data["pca_17"]) * (np.where(0.636620 < -9998, data["pca_37"], (-1.0*((data["pca_37"]))) )))) / 2.0)))/2.0))))))) / 2.0)))/2.0)))/2.0)))) +

            0.999414*np.tanh(((((0.318310) * (np.where(data["pca_5"]>0, (((((np.where(data["pca_1"]>0, data["pca_2"], data["pca_5"] )) + (data["pca_1"]))/2.0)) / 2.0), np.where(data["pca_12"]>0, np.where(data["pca_5"]>0, 0.318310, (((data["pca_11"]) + (data["pca_7"]))/2.0) ), ((0.318310) / 2.0) ) )))) / 2.0)) +

            1.0*np.tanh(np.where(0.0 < -9998, 0.0, ((np.where(data["pca_4"]<0, 0.0, ((-1.0) / 2.0) )) / 2.0) )) +

            1.0*np.tanh(np.where(((((data["pca_46"]) - (1.570796))) - ((7.14942264556884766)))<0, (((-1.0*((((((0.318310) / 2.0)) / 2.0))))) * (data["pca_46"])), np.where(data["pca_46"]<0, ((0.318310) + (np.where(((((0.318310) / 2.0)) - (data["pca_46"]))<0, data["pca_46"], data["pca_46"] ))), (7.14942264556884766) ) )) +

            1.0*np.tanh((((((((((((data["pca_43"]) / 2.0)) * ((-1.0*((((((data["pca_46"]) / 2.0)) / 2.0))))))) / 2.0)) + (((((np.where(data["pca_2"]>0, data["pca_33"], data["pca_49"] )) / 2.0)) / 2.0)))/2.0)) / 2.0)) +

            0.901543*np.tanh(((np.where(data["pca_6"]>0, np.where(data["pca_6"]>0, np.where(data["pca_2"]<0, (-1.0*((((((((data["pca_7"]) + (data["pca_35"]))/2.0)) + ((((((data["pca_6"]) * (0.318310))) + (data["pca_34"]))/2.0)))/2.0)))), (((((data["pca_6"]) + (data["pca_34"]))/2.0)) / 2.0) ), 0.318310 ), 0.318310 )) / 2.0)) +

            0.999414*np.tanh((((-1.0*((((((0.29596573114395142)) + (((np.where(data["pca_5"]>0, 0.636620, ((np.where(data["pca_1"]>0, 0.636620, data["pca_32"] )) + (((np.where(data["pca_5"] < -9998, data["pca_5"], np.where(data["pca_4"]>0, (((data["pca_1"]) + (data["pca_4"]))/2.0), data["pca_33"] ) )) / 2.0))) )) / 2.0)))/2.0))))) / 2.0)) +

            0.962883*np.tanh(np.where(data["pca_6"]>0, (((-1.0*((np.where(data["pca_11"]<0, ((((data["pca_40"]) / 2.0)) / 2.0), np.where(data["pca_3"]<0, 1.0, np.where((((((data["pca_3"]) + (data["pca_38"]))/2.0)) / 2.0)<0, data["pca_11"], ((-1.0) / 2.0) ) ) ))))) / 2.0), np.where(data["pca_11"]<0, 0.318310, 0.0 ) )) +

            1.0*np.tanh((((((((((((np.where((-1.0*((data["pca_41"])))<0, np.where((-1.0*((data["pca_41"])))<0, data["pca_22"], data["pca_22"] ), ((data["pca_8"]) + (data["pca_30"])) )) + (np.where(data["pca_29"]<0, (-1.0*((np.where(data["pca_37"]>0, data["pca_29"], data["pca_8"] )))), data["pca_4"] )))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(((np.where(data["pca_1"]>0, ((np.where(data["pca_0"]<0, ((((data["pca_40"]) - (((data["pca_1"]) - (data["pca_7"]))))) / 2.0), (((-1.0*((((np.where(data["pca_7"]>0, data["pca_1"], data["pca_0"] )) - (((data["pca_42"]) + (data["pca_1"])))))))) * 2.0) )) / 2.0), 0.0 )) / 2.0)) +

            0.878101*np.tanh(np.where(((((1.0) / 2.0)) * (np.where(data["pca_0"]<0, data["pca_1"], (((1.0) + (np.where(((0.0) + (0.0))<0, 1.0, (((0.636620) + ((-1.0*((((data["pca_0"]) / 2.0))))))/2.0) )))/2.0) )))<0, (-1.0*((((((1.0) / 2.0)) / 2.0)))), 1.0 )) +

            1.0*np.tanh(((((((((((((((((np.where(data["pca_22"]>0, data["pca_6"], data["pca_46"] )) + (((data["pca_46"]) + (data["pca_35"]))))) + (np.where(data["pca_19"]>0, data["pca_6"], ((data["pca_40"]) + (data["pca_40"])) )))/2.0)) + (data["pca_40"]))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh((((-1.0*(((((((((((np.where(data["pca_40"]>0, data["pca_7"], np.where(data["pca_29"]<0, data["pca_29"], np.where(((data["pca_36"]) / 2.0)<0, data["pca_9"], ((data["pca_21"]) / 2.0) ) ) )) + (((data["pca_21"]) / 2.0)))/2.0)) / 2.0)) / 2.0)) / 2.0))))) / 2.0)) +

            1.0*np.tanh((((((((((np.where((((data["pca_4"]) + (data["pca_0"]))/2.0)>0, (-1.0*((data["pca_15"]))), -1.0 )) + (np.where(data["pca_15"] < -9998, data["pca_0"], np.where(data["pca_4"]>0, -1.0, np.where((((data["pca_4"]) + (data["pca_0"]))/2.0)>0, (-1.0*((data["pca_4"]))), data["pca_15"] ) ) )))/2.0)) / 2.0)) / 2.0)) / 2.0)) +

            0.999805*np.tanh(0.0) +

            0.932213*np.tanh(((0.318310) * (np.where(data["pca_0"]<0, (((((((data["pca_18"]) / 2.0)) + (0.0))/2.0)) / 2.0), ((((((-1.0*((data["pca_18"])))) / 2.0)) + (np.where(data["pca_1"]>0, data["pca_10"], (((data["pca_37"]) + ((((data["pca_12"]) + (data["pca_23"]))/2.0)))/2.0) )))/2.0) )))) +

            1.0*np.tanh(np.where(data["pca_5"]>0, 0.0, np.where(((np.where(0.318310<0, data["pca_19"], data["pca_19"] )) - (data["pca_6"]))<0, 0.0, 0.318310 ) )) +

            1.0*np.tanh(((np.where(data["pca_9"]<0, ((((data["pca_8"]) / 2.0)) * ((((0.0)) / 2.0))), (-1.0*((0.318310))) )) / 2.0)) +

            1.0*np.tanh((((((((0.0)) + (((((((((((((data["pca_28"]) * (((((((((data["pca_14"]) + (0.0))) / 2.0)) / 2.0)) / 2.0)))) + (((((data["pca_6"]) / 2.0)) / 2.0)))/2.0)) + ((-1.0*((0.0)))))/2.0)) / 2.0)) / 2.0)))/2.0)) + ((-1.0*((0.0)))))/2.0)) +

            0.848017*np.tanh((((0.0)) / 2.0)) +

            0.940223*np.tanh(((((((np.where(0.0<0, np.where(0.0<0, 0.0, np.where(0.0 < -9998, 0.0, 0.0 ) ), 0.0 )) / 2.0)) * 2.0)) * 2.0)) +

            1.0*np.tanh((((((((0.00239253090694547)) * (data["pca_45"]))) * (data["pca_45"]))) * ((-1.0*((((np.where(data["pca_8"]<0, data["pca_10"], data["pca_14"] )) + (((data["pca_8"]) + (np.where((0.00239253090694547)<0, (((((0.00239253090694547)) * ((0.00239253090694547)))) * (data["pca_45"])), data["pca_37"] ))))))))))) +

            0.890018*np.tanh(((0.318310) * (((((np.where(data["pca_32"]>0, ((np.where(data["pca_14"]>0, data["pca_35"], 0.318310 )) / 2.0), ((np.where(data["pca_35"]<0, data["pca_6"], data["pca_32"] )) / 2.0) )) / 2.0)) / 2.0)))) +

            0.971283*np.tanh((((0.01871824637055397)) * (np.where(data["pca_9"]<0, ((data["pca_17"]) - (data["pca_9"])), ((((((((np.where(data["pca_38"]>0, np.where(data["pca_28"]>0, data["pca_49"], data["pca_40"] ), data["pca_39"] )) * (data["pca_9"]))) - (data["pca_42"]))) - (data["pca_28"]))) - (data["pca_38"])) )))) +

            0.841375*np.tanh(0.0) +

            0.999609*np.tanh(np.where((-1.0*((((data["pca_6"]) - (data["pca_45"])))))>0, ((0.318310) / 2.0), np.where(np.where(((0.318310) / 2.0) < -9998, (-1.0*(((-1.0*((((data["pca_6"]) - (0.318310)))))))), (-1.0*((0.318310))) ) < -9998, 0.318310, (((-1.0*((0.318310)))) / 2.0) ) )) +

            1.0*np.tanh(((np.where((((10.04058361053466797)) + (((((((((data["pca_4"]) / 2.0)) / 2.0)) + (data["pca_4"]))) / 2.0)))>0, ((((((((data["pca_46"]) / 2.0)) * (((((((data["pca_4"]) / 2.0)) / 2.0)) / 2.0)))) / 2.0)) / 2.0), ((((((data["pca_1"]) + (data["pca_4"]))) / 2.0)) / 2.0) )) / 2.0)) +

            1.0*np.tanh((((((((((data["pca_47"]) + (((((((data["pca_20"]) + (data["pca_4"]))/2.0)) + ((((data["pca_12"]) + (np.where(data["pca_5"]<0, data["pca_1"], data["pca_12"] )))/2.0)))/2.0)))/2.0)) / 2.0)) * (np.where(0.0>0, (0.06947280466556549), (((0.06947280466556549)) * (data["pca_6"])) )))) / 2.0)) +

            0.879078*np.tanh(np.where(((0.318310) - (data["pca_0"]))>0, np.where((((((data["pca_2"]) + (data["pca_3"]))/2.0)) - ((-1.0*(((((((0.0) + (0.318310))/2.0)) - ((-1.0*((data["pca_4"]))))))))))>0, 0.0, 0.318310 ), (((-1.0*((data["pca_17"])))) * ((0.04154564067721367))) )) +

            0.998047*np.tanh(((((((0.318310) * (np.where(((-1.0) * (data["pca_8"]))<0, ((data["pca_34"]) / 2.0), ((((-1.0*((((data["pca_7"]) / 2.0))))) + (((((-1.0*((data["pca_10"])))) + (((((((data["pca_8"]) + (((-1.0) * (data["pca_10"]))))/2.0)) + (data["pca_8"]))/2.0)))/2.0)))/2.0) )))) / 2.0)) / 2.0)) +

            1.0*np.tanh((((((((((((((data["pca_13"]) + ((((((((data["pca_37"]) * (data["pca_33"]))) + (((data["pca_12"]) * ((((((data["pca_12"]) * (0.318310))) + (data["pca_33"]))/2.0)))))) + ((-1.0*((data["pca_11"])))))/2.0)))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            0.999414*np.tanh(((((np.where(((data["pca_28"]) * 2.0)>0, 0.0, (-1.0*((np.where(3.141593 < -9998, 0.636620, (((-2.0) + (np.where(data["pca_37"]>0, 3.141593, np.where(data["pca_9"]>0, data["pca_47"], 3.141593 ) )))/2.0) )))) )) / 2.0)) / 2.0)) +

            1.0*np.tanh(((((((((((((((data["pca_47"]) + (((((-1.0*((data["pca_9"])))) + (data["pca_45"]))/2.0)))/2.0)) + ((((np.where(data["pca_37"]>0, data["pca_10"], data["pca_47"] )) + ((((((data["pca_47"]) / 2.0)) + (data["pca_47"]))/2.0)))/2.0)))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(((data["pca_41"]) * ((((-1.0*((np.where(((((0.0) / 2.0)) + (data["pca_40"]))<0, np.where(np.where(((((0.0) / 2.0)) / 2.0)<0, 0.0, ((0.318310) / 2.0) ) < -9998, data["pca_41"], 0.0 ), ((((0.318310) / 2.0)) / 2.0) ))))) / 2.0)))) +

            0.953702*np.tanh((((((((-1.0*((0.0)))) / 2.0)) / 2.0)) / 2.0)) +

            0.916976*np.tanh(0.0) +

            1.0*np.tanh((((-1.0*((np.where((((5.0)) - ((((data["pca_33"]) + (1.570796))/2.0)))<0, ((((data["pca_48"]) - ((((5.0)) * (0.0))))) + ((-1.0*((((data["pca_2"]) - (data["pca_48"]))))))), (((((5.0)) * 2.0)) * (0.0)) ))))) / 2.0)) +

            1.0*np.tanh(((((((((-1.0*((((((data["pca_43"]) / 2.0)) / 2.0))))) / 2.0)) + (((((np.where(((data["pca_1"]) * (((((data["pca_43"]) / 2.0)) / 2.0)))>0, ((((data["pca_15"]) / 2.0)) / 2.0), (-1.0*((((data["pca_15"]) / 2.0)))) )) / 2.0)) * (((((data["pca_43"]) / 2.0)) / 2.0)))))/2.0)) / 2.0)) +

            0.952530*np.tanh(0.0) +

            1.0*np.tanh(0.0) +

            1.0*np.tanh(np.where((((-1.0*((((np.where(data["pca_1"]<0, data["pca_0"], np.where((((13.03970813751220703)) + (data["pca_0"])) < -9998, (0.0), ((data["pca_1"]) + ((-1.0*(((((13.03971195220947266)) + (data["pca_0"]))))))) ) )) / 2.0))))) + ((13.03971195220947266)))<0, (((13.03971195220947266)) + (data["pca_0"])), 0.0 )) +

            0.976167*np.tanh(((np.where(data["pca_17"]>0, (0.12955072522163391), ((((((data["pca_38"]) + ((-1.0*((np.where((0.12955072522163391)>0, data["pca_0"], data["pca_38"] ))))))/2.0)) + (np.where((-1.0*((np.where(data["pca_4"]>0, data["pca_0"], (-1.0*((data["pca_1"]))) ))))>0, data["pca_1"], data["pca_38"] )))/2.0) )) * ((((0.12955072522163391)) / 2.0)))) +

            0.999805*np.tanh(0.0) +

            0.816761*np.tanh((-1.0*((((np.where(((data["pca_1"]) * ((((((((((((data["pca_2"]) + (data["pca_5"]))/2.0)) + (data["pca_2"]))/2.0)) + ((-1.0*(((((((0.318310) + (data["pca_6"]))/2.0)) / 2.0))))))/2.0)) * 2.0)))<0, 0.318310, (((-1.0*((np.where(data["pca_1"]<0, 0.318310, data["pca_40"] ))))) * 2.0) )) / 2.0))))) +

            1.0*np.tanh(0.0) +

            1.0*np.tanh((((((((((((np.where(data["pca_12"]<0, data["pca_17"], ((((((data["pca_1"]) / 2.0)) / 2.0)) / 2.0) )) + (np.where(data["pca_10"]<0, 0.0, (-1.0*(((((data["pca_40"]) + (((data["pca_28"]) * (((((data["pca_1"]) / 2.0)) / 2.0)))))/2.0)))) )))/2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(((data["pca_10"]) * ((-1.0*((((((((2.97544312477111816)) + (-3.0))/2.0)) * (np.where(np.where(data["pca_10"]>0, data["pca_1"], data["pca_1"] )>0, ((((2.97543954849243164)) + (data["pca_28"]))/2.0), np.where(data["pca_8"]<0, -3.0, np.where((((data["pca_9"]) + (-3.0))/2.0)<0, (2.97543954849243164), data["pca_28"] ) ) ))))))))) +

            0.888259*np.tanh(np.where(((((((0.318310) - (((data["pca_4"]) / 2.0)))) / 2.0)) - (data["pca_31"]))<0, 0.0, np.where(((data["pca_4"]) / 2.0)<0, np.where(((data["pca_4"]) / 2.0)<0, (((0.318310) + (((((((data["pca_4"]) / 2.0)) * (0.318310))) / 2.0)))/2.0), 0.318310 ), 0.318310 ) )) +

            1.0*np.tanh(np.where((0.05075455829501152) < -9998, (0.05075455829501152), ((((((((((((data["pca_8"]) - (((data["pca_3"]) * 2.0)))) * 2.0)) / 2.0)) / 2.0)) / 2.0)) * ((((0.05075098201632500)) / 2.0))) )) +

            1.0*np.tanh(0.0) +

            0.999609*np.tanh(((np.where(((0.636620) - ((-1.0*((((data["pca_22"]) - (data["pca_1"])))))))>0, np.where(np.where(data["pca_16"] < -9998, 0.0, data["pca_16"] )>0, 0.0, np.where((-1.0*((data["pca_8"])))>0, np.where(data["pca_16"]>0, 0.0, 0.636620 ), 0.0 ) ), 0.0 )) / 2.0)) +

            0.999805*np.tanh(((np.where(((data["pca_40"]) / 2.0) < -9998, 0.318310, ((((np.where((((((data["pca_0"]) + (data["pca_4"]))/2.0)) / 2.0)<0, 0.318310, (-1.0*((((((-1.0*((data["pca_45"])))) + (((data["pca_25"]) / 2.0)))/2.0)))) )) / 2.0)) / 2.0) )) / 2.0)) +

            0.917367*np.tanh(np.where(((((data["pca_24"]) * 2.0)) * (data["pca_29"]))<0, 0.0, np.where(0.318310 < -9998, ((0.318310) / 2.0), (-1.0*((((0.318310) / 2.0)))) ) )) +

            0.902911*np.tanh(((((0.318310) * (0.0))) / 2.0)) +

            1.0*np.tanh(((((((np.where(data["pca_41"]>0, np.where(((data["pca_1"]) * (data["pca_23"])) < -9998, 0.0, (((((((((data["pca_6"]) + (((data["pca_1"]) + (0.318310))))/2.0)) / 2.0)) / 2.0)) / 2.0) ), 0.318310 )) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(((0.318310) * (((((data["pca_27"]) * (((np.where(np.where(data["pca_27"] < -9998, (-1.0*(((((-1.0*((data["pca_0"])))) / 2.0)))), (-1.0*((data["pca_0"]))) )>0, 0.318310, np.where(data["pca_5"]>0, (0.0), (-1.0*((((0.318310) * 2.0)))) ) )) / 2.0)))) / 2.0)))) +

            0.976363*np.tanh((0.00827551074326038)) +

            1.0*np.tanh(0.0) +

            1.0*np.tanh((((((0.0) + (((np.where(data["pca_38"]>0, ((((((data["pca_28"]) + (data["pca_4"]))/2.0)) + (np.where((((data["pca_4"]) + (data["pca_38"]))/2.0)<0, data["pca_28"], (((data["pca_17"]) + (data["pca_4"]))/2.0) )))/2.0), (((0.0) + (data["pca_7"]))/2.0) )) * (((((2.70234060287475586)) + (-3.0))/2.0)))))/2.0)) / 2.0)) +

            0.848017*np.tanh(((((((np.where(data["pca_8"]<0, (((-1.0*((((data["pca_46"]) / 2.0))))) / 2.0), np.where(data["pca_0"]>0, data["pca_46"], ((((((0.318310) * (data["pca_0"]))) / 2.0)) * (0.318310)) ) )) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(np.where(data["pca_2"] < -9998, (0.0), np.where(np.where(data["pca_4"]<0, data["pca_3"], data["pca_8"] )<0, 0.0, np.where(((((-1.0*((((data["pca_3"]) / 2.0))))) + (((data["pca_0"]) - (np.where(data["pca_2"]<0, data["pca_1"], data["pca_8"] )))))/2.0)<0, 0.318310, (((-1.0*((0.318310)))) * 2.0) ) ) )) +

            0.845282*np.tanh(np.where(((((np.where(data["pca_3"] < -9998, 0.318310, 0.318310 )) * (((0.318310) * (data["pca_2"]))))) * (data["pca_3"]))>0, np.where(data["pca_4"]<0, 0.318310, ((0.318310) / 2.0) ), np.where(data["pca_4"]<0, (-1.0*((0.318310))), ((((0.318310) / 2.0)) / 2.0) ) )) +

            1.0*np.tanh(((((np.where(((data["pca_17"]) / 2.0)>0, ((np.where(data["pca_1"]>0, ((np.where(((((data["pca_17"]) * 2.0)) - (data["pca_1"]))<0, -2.0, data["pca_7"] )) / 2.0), ((((np.where((-1.0*((data["pca_33"])))<0, data["pca_5"], data["pca_33"] )) / 2.0)) / 2.0) )) / 2.0), 0.0 )) / 2.0)) / 2.0)) +

            1.0*np.tanh(((((np.where(data["pca_28"] < -9998, data["pca_3"], np.where(data["pca_0"]<0, ((np.where(data["pca_28"]>0, np.where(data["pca_4"]<0, np.where(data["pca_3"]<0, ((data["pca_5"]) / 2.0), data["pca_3"] ), ((-2.0) / 2.0) ), np.where(data["pca_4"]>0, -2.0, (0.0) ) )) / 2.0), (0.58171403408050537) ) )) / 2.0)) / 2.0)) +

            1.0*np.tanh((-1.0*((np.where((0.0)<0, 0.0, np.where(((((((((((0.318310) - (data["pca_2"]))) - (data["pca_2"]))) - (data["pca_2"]))) - (data["pca_7"]))) - (data["pca_3"]))<0, np.where(data["pca_2"]<0, data["pca_5"], 0.0 ), 0.0 ) ))))) +

            0.999609*np.tanh(((((((np.where(np.where(((data["pca_6"]) * 2.0)<0, data["pca_7"], ((data["pca_4"]) / 2.0) )<0, np.where(data["pca_16"]<0, data["pca_35"], ((data["pca_24"]) / 2.0) ), (((((-1.0*((data["pca_1"])))) / 2.0)) / 2.0) )) / 2.0)) / 2.0)) * (((0.318310) / 2.0)))) +

            0.999609*np.tanh(((((0.0) * ((((0.07969381660223007)) * (np.where((0.07969024032354355)>0, (-1.0*((np.where(0.0>0, (0.0), 0.318310 )))), data["pca_45"] )))))) * ((-1.0*((0.0)))))) +

            1.0*np.tanh(((((((np.where(data["pca_40"]>0, ((np.where(data["pca_4"]>0, ((((np.where(data["pca_8"]>0, (-1.0*((data["pca_40"]))), data["pca_40"] )) / 2.0)) / 2.0), (-1.0*((data["pca_40"]))) )) / 2.0), ((((np.where(data["pca_38"]>0, data["pca_8"], (-1.0*((data["pca_29"]))) )) / 2.0)) / 2.0) )) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh((0.0)) +

            0.999805*np.tanh(0.0))



plt.figure(figsize=(15,15))

_ = plt.scatter( GPx(gpdata),

                 GPy(gpdata),

                 s=1)