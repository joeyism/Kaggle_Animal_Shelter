from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation

import numpy as np
import pandas

animals = pandas.read_csv("data/train.csv")
animals_test = pandas.read_csv("data/test.csv")

# Maps
animalTypeMapping = {"Dog": 0, "Cat": 1}
outcomeTypeMapping = {'Return_to_owner':0, 'Euthanasia':1, 'Adoption':2, 'Transfer':3, 'Died':4 }
genderMapping = {"Male": 0, "Female": 1, "Unknown": 2}
reproductiveMapping = {"Intact": 0, "Neutered": 1, "Spayed": 2, "Unknown":3}
timeMapping = { "day": 1, "days":1, "week": 7, "weeks":7, "month": 30, "months":30, "year": 365, "years": 365 }
outcomeSubtypeMapping = {'Suffering':1, 'Foster':2, 'Partner':3, 'Offsite':4, 'SCRP':5, 'Aggressive':6, 'Behavior':7, 'Rabies Risk':8, 'Medical':9, 'In Kennel':10, 'In Foster':11, 'Barn':12, 'Court/Investigation':13, 'Enroute':14, 'At Vet':15, 'In Surgery':16  }
breedMapping = {"Chihuahua": 1, "Beagle": 2, "Labrador": 3, "Domestic Shorthair": 4, "Pit Bull": 5, "Dachshund": 6, "Border Collie":7, "Shepherd":8 , "Terrier":9, "Retriever": 10, "Dane": 11, "Sheepdog": 12, "Hound": 13, "Domestic Longhair":14, "Domestic Medium Hair":15, "Boxer":16}

# Predictors
methods = ["Random Forest", "Gradient Boosting", "Extra Trees Classifier"]
randomForestPredictors = ["AnimalType", "Gender", "Reproductive", "Age", "Mix", "HasName", "IsDay"]

def get_gender(title):
    genderArr = title.split()
    if len(genderArr) == 0 or genderArr[0] == "Unknown":
        return 2
    genderType = genderArr[1]
    return genderMapping[genderType] 

def get_reproductive(title):
    genderArr = title.split()
    if len(genderArr) == 0 or genderArr[0] == "Unknown":
        return 3
    reproductiveType = genderArr[0]
    return reproductiveMapping[reproductiveType]
    
def get_age(title):
    ageArr = title.split()
    ageNo = int(ageArr[0])
    timeSegment = ageArr[1]
    return ageNo*timeMapping[timeSegment]

def get_mix(title):
    breedArr = title.split()
    mixOrNot = breedArr[len(breedArr)-1]
    return 1 if mixOrNot == "Mix" else  0

def get_mix_colours(title): # mix colours: 0 false, 1 true
    return 1 if title.find("/") > -1 else 0

def has_name(title): # 0 if no name, 1 if has name
    return 0 if pandas.isnull(title) else 1

def is_day(title): # 1 if it happens between 8am - 10 pm, 0 otherwise
    dayBegins=8
    dayEnds=22
    time= title.split()[1]
    hour = int(time.split(":")[0])
    return 1 if hour < dayEnds and hour > dayBegins else 0

def map_data(data):
    for k,v in animalTypeMapping.items():
        data.loc[data["AnimalType"] == k, "AnimalType"] = v
    data["SexuponOutcome"] = data["SexuponOutcome"].fillna("Unknown")
    data["Gender"] = data["SexuponOutcome"].apply(get_gender)
    data["Reproductive"] = data["SexuponOutcome"].apply(get_reproductive)
    data["Age"] = data["AgeuponOutcome"].fillna("0 day")
    data["Age"] = data["Age"].apply(get_age)
    data.loc[data["Age"] == 0, "Age"] = data["Age"].median()
    data["Mix"] = data["Breed"].apply(get_mix)
    data["MixedColours"] = data["Color"].apply(get_mix_colours)
    data["HasName"] = data["Name"].apply(has_name)
    data["IsDay"] = data["DateTime"].apply(is_day)
    for k,v in breedMapping.items():
        data.loc[data["Breed"].str.contains(k), "BreedType"] = v
    data["BreedType"] = data["BreedType"].fillna(0)
    return data

def expandOutcome(data):
    result = pandas.DataFrame(index=data["AnimalID"], columns=outcomeTypeMapping.keys())
    result = result.fillna(0)
    for k,v in outcomeTypeMapping.items():
        Matches = (data["OutcomeType"] == v)[:]
        result[k][Matches[Matches].index] = 1
    return result

def map_outcome(data):
    for k,v in outcomeTypeMapping.items():
        data.loc[data["OutcomeType"] == k, "OutcomeType"] = v
    return data

def binarize(data):
    result = data.copy()
    result.loc[:,:]=0
    for index, row in data.iterrows():
        maxValue = np.amax(row)
        maxIndex = row[row == maxValue].index[0]
        result.set_value(index, maxIndex, 1)
    return result 

def normalize(data):
    result = data.copy()
    for index, row in data.iterrows():
        sumValue = np.sum(row)
        normalizedRow = row / sumValue
        result.loc[index, :] = normalizedRow
    return result 

def submission(idFrame, data, name):
    result = pandas.concat([idFrame, data], axis=1)
    print("Writing to " + name + ".csv")
    result.to_csv(name+".csv", index=False)

# Mapping
print("Mapping data")
animals = map_data(animals)
animals.to_csv("modified_animals.csv", index=False)
print("Mapping outcome")
animals = map_outcome(animals)
outcome = expandOutcome(animals)
print("Mapping test data")
animals_test = map_data(animals_test)

# Ensemble
print("Calculating Stacked Methods")
algorithms = [
        [
            RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=5, min_samples_leaf=1),
            randomForestPredictors
            ],
        [
            GradientBoostingClassifier(random_state=1, n_estimators=200, max_depth=3),
            randomForestPredictors
            ],
        [
            ExtraTreesClassifier(random_state=1, n_estimators=500, min_samples_split=5, min_samples_leaf=1),
            randomForestPredictors
            ]
        ]

full_predictions = []
overall_mean = []
i = 0;
for alg, predictors in algorithms:
    this_predictions = pandas.DataFrame(index=animals_test.index.values, columns=outcomeTypeMapping.keys())
    this_predictions = this_predictions.fillna(0)
    mean_score = 0
    for columns in this_predictions:
        scores = cross_validation.cross_val_score(alg, animals[predictors], outcome[columns].astype(float), cv=3)
        print(methods[i] + " for "+columns+"'s accuracy is " + str(scores.mean()))
        mean_score +=scores.mean()
        alg.fit(animals[predictors], outcome[columns])
        predictions = alg.predict_proba(animals_test[predictors].astype(float))[:,1]
        this_predictions[columns]=predictions
    mean_score /= len(this_predictions.columns)
    print(methods[i] + "'s mean score is " + str(mean_score) + "\n")
    full_predictions.append(this_predictions)
    overall_mean.append(mean_score)
    i = i+1

idFrame = pandas.DataFrame({"ID":animals_test["ID"]})


combined_predictions = (full_predictions[0] + full_predictions[1] + full_predictions[2])/3
avgMean = (overall_mean[0] + overall_mean[1] + overall_mean[2])/3
print("Overall mean is "+str(avgMean) +"\n")

for i, predictions in enumerate(full_predictions):
#    full_predictions[i] = binarize(predictions)
    full_predictions[i] = normalize(predictions)
    submission(idFrame, full_predictions[i], "full_predictions_"+methods[i])
#combined_predictions = binarize(combined_predictions)
combined_predictions = normalize(combined_predictions)
submission(idFrame, combined_predictions, "combined_predictions_avg_"+str(avgMean)) 

print("Done")
