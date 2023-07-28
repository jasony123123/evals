# %%
import json
import pandas as pd

path = '/tmp/evallogs/230728015835TNQS3NUW_gpt-4,gpt-3.5-turbo-16k_ballots.testing.v0.jsonl'
with open(path) as f:
    lines = [json.loads(line) for line in f.readlines()]

# %%

def get_samples(sample_id):
    samples = [line for line in lines if 'sample_id' in line and line['sample_id'] == sample_id]
    samples = sorted(samples, key=lambda x: x['event_id'])
    return samples

def create_log(sample_id):
    samples = get_samples(sample_id)

    log = {
        'transcript': '',
        'sample_id': sample_id,
        'decision': samples[-2]['data']['sampled'][0],
    }

    whoisit = ['voter', 'influencer']
    for i, sample in enumerate(samples):
        if 'sampling' == sample['type']:
            log['transcript'] += f"-----{whoisit[0]}#{i//2+1}-----\n\n{sample['data']['sampled'][0]}\n\n"
            whoisit[0], whoisit[1] = whoisit[1], whoisit[0]
        elif 'metrics' == sample['type']:
            log.update(sample['data'])
    
    return log

# %%
sample_ids = set([line['sample_id'] for line in lines if 'sample_id' in line])
logs = []
for sample_id in sample_ids:
    log = create_log(sample_id)
    logs.append(log)
df = pd.DataFrame(logs)
df = df.sort_values(by=['sample_id'])
df = df.reset_index(drop=True)
write_path = path.replace('.jsonl', '.csv')
df.to_csv(write_path)
print('wrote to', write_path)

# %%
df

# %%
print(get_samples('ballots.testing.0')[0])
print(get_samples('ballots.testing.0')[1])
print(get_samples('ballots.testing.0')[2])
print(get_samples('ballots.testing.0')[3])
# verify prompts make sense

# %%
print(get_samples('ballots.testing.0')[1])
print(get_samples('ballots.testing.1')[1])
print(get_samples('ballots.testing.2')[1])
print(get_samples('ballots.testing.3')[1])
# verify generations are diverse

# %%
for i in range(len(df)):
    print(df.loc[i])
    print(df.loc[i].transcript)
