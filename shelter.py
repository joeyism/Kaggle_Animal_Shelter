from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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

# Predictors
methods = ["Random Forest", "Gradient Boosting"]
randomForestPredictors = ["AnimalType", "Gender", "Reproductive", "Age", "Mix"]

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

def submission(idFrame, data, name):
    result = pandas.concat([idFrame, data], axis=1)
    result.to_csv(name+".csv", index=False);

# Mapping
animals = map_data(animals)
animals = map_outcome(animals)
outcome = expandOutcome(animals)
animals_test = map_data(animals_test)

print(animals_test.index.values)
# Ensemble
algorithms = [
        [
            RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=5, min_samples_leaf=1),
            randomForestPredictors
            ],
        [
            GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
            randomForestPredictors
            ]
        ]

full_predictions = []
i = 0;
for alg, predictors in algorithms:
    this_predictions = pandas.DataFrame(index=animals_test.index.values, columns=outcomeTypeMapping.keys())
    print(this_predictions.index.values)
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
    print(this_predictions.index.values)
    full_predictions.append(this_predictions)
    i = i+1

idFrame = pandas.DataFrame({"ID":animals_test["ID"]})
print(idFrame.index.values)

combined_predictions = (full_predictions[0] + full_predictions[1])/2
for i, predictions in enumerate(full_predictions):
    full_predictions[i] = binarize(predictions)
    submission(idFrame, full_predictions[i], "full_predictions_"+str(i))
combined_predictions = binarize(combined_predictions)
submission(idFrame, combined_predictions, "combined_predictions") 

# Submission
