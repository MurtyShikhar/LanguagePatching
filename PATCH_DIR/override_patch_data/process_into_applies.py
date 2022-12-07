import json

DEFAULT_EXP='default noop explanation'
def get_cond(explanation):
    return ' '.join(explanation.split(',')[0].split(' ')[1:])

with open('synthetic_data_old.json', 'r') as reader:
    data = json.load(reader)



new_data = {key: [] for key in data}
all_explanations = set()
for exp, instance, label in zip(data['explanations'], data['instances'], data['labels']):
    if exp == DEFAULT_EXP:
        new_data['labels'].append(0) # this is because we give all negative explanations the label of the noop...
        new_data['instances'].append(instance)
        new_data['explanations'].append(exp)
    else:
        new_data['labels'].append(1)
        cond = get_cond(exp)
        new_data['instances'].append(instance)
        new_data['explanations'].append(cond)

    all_explanations.add(cond)

for exp in all_explanations:
    print(exp)

with open('synthetic_data.json', 'w') as writer:
    json.dump(new_data,writer)
print(len(new_data['explanations']))
