import torch
from torchaudio import transforms as aT

import pandas as pd
localisation = pd.read_csv("chemin/vers/ton_fichier.csv")  # Remplace par le vrai chemin

cp = 0
a= 0
for i,l in localisation.iterrows():
    file = l['Fichier']
    loc = float(l['Localisation'])
    sub_fichier = fichier[fichier['Fichier']==file]
    sub_loc = sub_fichier[sub_fichier['loc_min'] <= loc]
    sub_loc = sub_loc[sub_loc['loc_max'] >= loc]
    if len(sub_loc) <1:
        print("File not found in localisation: ", file, "at location", loc)
    elif len(sub_loc) > 1:
        print("Multiple locations found for file: ", file)
        print(sub_loc[['loc_min', 'loc_max']])
    else :
        f = file[:-4]+"_loc"+str(sub_loc['loc_min'].values[0])+'-'+str(sub_loc['loc_max'].values[0])+'s_w60s.wav'
        if not os.path.exists(os.path.join(dir, f)):
            pattern = make_regex_pattern(file[:-4])
            files = [f for f in os.listdir(dir) if pattern.match(f)]
            if len(files) == 0:
                pass
                print(f"File not found for pattern: {file}")
                a+=1
                print('----------------------')
            else :
                cp = 0
                for filename in files:
                    # filename = "{file}_loc2773.997559-2833.997559s_w60s.wav"
                    # Expression régulière pour capturer deux floats après "_loc"
                    match = re.search(r"_loc(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", filename)
                    if match:
                        start = float(match.group(1))
                        end = float(match.group(2))
                        if start <= loc <= end:
                            cp = 1
                            break
                if cp == 0:
                    a+=1
                    print(file, "at location", loc)
                    print("disponible : ", [fil.split('_loc')[1][:-10] for fil in files])
                    # print(localisation[localisation['Fichier']==file]['Localisation'].unique())
                    #print(f"File found but not matching location: {loc}")
                    # print(f"Matching files: {files}")
                    print("----------------------------------------")
            # cp+=1
            # print(file)
            # print("====================")
print(a)

